"""
Match one user's questionnaire answers to restaurants with a support vector
machine (SVM) on a cosine kernel, and return the top 5.

Why an SVM on top of cosine similarity
--------------------------------------
Restaurant vectors and taste profiles are unit length, so cosine similarity is
just a dot product. Plain cosine ranking (score_restaurants in taste_profile.py)
has a weakness (EXPERIMENTS.md #3): every restaurant vector shares a large common
component, and a "dislikes" vector lowers every score by about the same amount,
so it barely changes the ranking.

An SVM learns from contrast instead (an "exemplar SVM"). For each person:
  positives  their likes vector + the restaurants with the best cosine score
             (likes minus dislikes, as in score_restaurants)
  negatives  their dislikes vector + the restaurants closest to it
             + a random sample of restaurants well below their top matches
The SVM uses a cosine kernel, so it still lives in cosine geometry, but its
decision boundary tilts away from what every restaurant shares and away from what
the person avoids. Restaurants are ranked by decision_function (distance from the
boundary), highest first.

Then the same hard rules as the rest of the project are applied: price range,
"must have" diets, and one
location per chain.

People who skipped (nothing to learn from) get the most popular well-rated
restaurants that fit their filters, the same fallback group_vector.py uses.

Covering every pick
-------------------
Someone who taps "Chinese" and "Indian" gets one averaged taste vector, and the
stronger signal can take over all five results (EXPERIMENTS.md #9, #11). So after
ranking, the top 5 are chosen so each tapped bubble is represented where possible
(like the coverage re-ranking in #12):
  1. Only restaurants in the better-scored half (COVER_SCORE_FRAC) are eligible, so
     coverage never reaches for a place the model already thinks is a poor fit.
  2. Each bubble gets its OWN candidates: the COVER_TOP eligible restaurants whose
     vectors are most similar to that bubble's description, and only if they stand
     out from the rest (similarity at least COVER_Z standard deviations above the
     eligible average). A bubble nobody matches gets no candidates and no label.
  3. Greedy: each round takes the restaurant that covers the most still-uncovered
     bubbles; ties and leftover slots go to the higher score.
This uses only the embeddings, never the Yelp cuisine tags (the experiments grade
with those). It only applies to people with 2+ tapped bubbles; pass cover=False
(or --no-cover) to turn it off. The `covers` column shows which bubbles each
result represents.

Run from the repo's top folder:
  python -m model.svm_recommend --user U0002
  python -m model.svm_recommend --limit 5 --method cosine     # plain cosine baseline
  python -m model.svm_recommend --user U0002 --no-cover       # no coverage step
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

from model.taste_profile import (BUBBLE_DESCRIPTIONS, DIET_CATEGORIES, allowed,
                                 build_taste_profile, get_model, score_restaurants)

CITY = "philadelphia"
VECTORS = Path("data") / "output" / "restaurant_vectors.parquet"
USERS = Path("data") / "mock" / "mock_users.json"

TOP_K = 5
SVM_C = 10.0              # margin softness; higher = fit the labels more tightly
PSEUDO_POSITIVES = 10     # best cosine-scored restaurants, used as extra positives
PSEUDO_NEGATIVES = 10     # restaurants closest to the dislikes vector, used as extra negatives
NEGATIVE_SAMPLES = 300    # random "everything else" negatives
NEGATIVE_SKIP = 0.20      # never sample negatives from the top 20% by cosine score
KERNEL = "rbf"         # "cosine" (linear on unit vectors) or "rbf" = exp(-RBF_GAMMA * (1 - cosine))
RBF_GAMMA = 10.0          # how local the "rbf" kernel is; higher = tighter clusters
PER_PICK_POSITIVES = 10    # >0: also train on this many positives for EACH tapped bubble (see below)
DIET_PREFER_BONUS = 0.0   # added to the SVM score of places tagged with a diet the user "prefers".
                          # 0 = off. The SVM's own score has no diet term (cosine's does), so
                          # without this a "prefer vegan" answer barely changes the results.
                          # Pick a value with: python -m model.tune_svm --diet-bonus
COVER_SCORE_FRAC = 0.5    # only the better-scored half of restaurants can fill a coverage slot
COVER_TOP = 15            # each tapped bubble gets its 15 most similar eligible restaurants
COVER_Z = 2.0             # ...and they must stand out: similarity >= mean + 2 std
SEED = 42


def profile_from_user(user, restaurants):
    """Turn one mock_users.json entry into a taste profile (see taste_profile.py).

    Tapped bubbles (cravings + specific) are embedded one by one, as decided in
    EXPERIMENTS.md #11/#13. A mock user's loves_text is only those bubbles joined
    into one string, so passing both would count them twice; it is used only when
    the person tapped no bubbles.
    `restaurants` must be the full restaurant_vectors table with its default index.
    """
    bubbles = [*user.get("cravings", []), *user.get("specific", [])]
    return build_taste_profile(
        restaurants,
        loves_text="" if bubbles else user.get("loves_text", ""),
        loved_names=user.get("loved_restaurants", []),
        dislikes_text=user.get("dislikes_text", ""),
        prices=user.get("prices", []),
        diets=user.get("diets", {}),
        bubbles=bubbles,
    )


def pick_vectors(picks):
    """One unit vector per tapped bubble (its description, embedded)."""
    return np.vstack([get_model().encode(BUBBLE_DESCRIPTIONS.get(p, p), normalize_embeddings=True)
                      for p in picks])


def _kernel(A, B):
    """Similarity between rows of A and B. Both options depend only on cosine."""
    cos = cosine_similarity(A, B)
    return cos if KERNEL == "cosine" else np.exp(-RBF_GAMMA * (1.0 - cos))


def svm_scores(V, base_score, likes, dislikes=None, pick_vecs=None, seed=SEED):
    """One SVM score per restaurant (higher = better match).

    V: (n, d) unit-length restaurant vectors. likes / dislikes: unit-length vectors.
    base_score: the plain cosine score per restaurant (score_restaurants); it picks
    the pseudo-labels, so the SVM starts from the baseline and sharpens it.
    pick_vecs: one vector per tapped bubble. With PER_PICK_POSITIVES > 0, each bubble
    adds its own positives (the restaurants most similar to that bubble, from the
    better-scored half), so a person who taps Chinese AND Indian is described by two
    clusters instead of one averaged vector that sits between them. That only helps
    with KERNEL = "rbf", which can draw a boundary around two separate clusters.
    """
    n = len(V)
    order = np.argsort(-base_score)                       # best cosine matches first
    pos = set(order[:min(PSEUDO_POSITIVES, n)].tolist())
    keep_out = set(order[:int(NEGATIVE_SKIP * n)].tolist())   # never sampled as negatives
    exemplars = [likes]
    if pick_vecs is not None and PER_PICK_POSITIVES > 0:
        better = order[:max(PER_PICK_POSITIVES, n // 2)]
        for pv in pick_vecs:
            pos |= set(better[np.argsort(-(V[better] @ pv))[:PER_PICK_POSITIVES]].tolist())
            keep_out |= set(np.argsort(-(V @ pv))[:int(NEGATIVE_SKIP * n)].tolist())
            exemplars.append(pv)
    pos_idx = np.array(sorted(pos))

    # Random negatives from everything except the top matches (those might be good
    # matches that just weren't picked as positives).
    pool = np.setdiff1d(np.arange(n), np.union1d(pos_idx, np.array(sorted(keep_out), dtype=int)))
    rng = np.random.default_rng(seed)
    neg_idx = rng.choice(pool, size=min(NEGATIVE_SAMPLES, len(pool)), replace=False)

    rows, labels = [np.vstack(exemplars), V[pos_idx]], [1] * (len(exemplars) + len(pos_idx))
    if dislikes is not None:
        near_dislikes = np.argsort(-(V @ dislikes))[:min(PSEUDO_NEGATIVES, n)]
        neg_idx = np.union1d(neg_idx, np.setdiff1d(near_dislikes, pos_idx))
        rows.append(dislikes[None, :])
        labels.append(0)
    rows.append(V[neg_idx])
    labels += [0] * len(neg_idx)

    if len(neg_idx) == 0:                                 # table too small to train on
        return base_score

    X, y = np.vstack(rows), np.array(labels)
    clf = SVC(kernel="precomputed", C=SVM_C, class_weight="balanced")
    clf.fit(_kernel(X, X), y)
    return clf.decision_function(_kernel(V, X))


def cover_top_k(ranked, V, picks, k, P=None):
    """Choose k rows of `ranked` so each tapped bubble in `picks` is represented.

    ranked: every allowed restaurant, sorted best score first, index = row positions
    in V. Returns the chosen rows (in score order) plus a `covers` column.
    """
    n = len(ranked)
    if n <= k:
        return ranked.assign(covers="")

    if P is None:
        P = pick_vectors(picks)
    sims = V[ranked.index.values] @ P.T                  # (n, picks): cosine to each bubble

    # Step 1: only the better-scored half can fill a coverage slot (rows are in score order).
    eligible = np.arange(min(n, max(k, int(COVER_SCORE_FRAC * n))))

    # Step 2: each bubble gets its own candidates among the eligible restaurants.
    covers = np.zeros((n, len(picks)), dtype=bool)
    for j in range(len(picks)):
        s = sims[eligible, j]
        floor = s.mean() + COVER_Z * s.std()              # must stand out from the rest
        best = eligible[np.argsort(-s)[:COVER_TOP]]
        covers[best[sims[best, j] >= floor], j] = True

    # Step 3: greedy. Most uncovered bubbles first; ties and leftovers by score.
    chosen, uncovered = [], np.ones(len(picks), dtype=bool)
    for _ in range(k):
        gain = (covers & uncovered).sum(axis=1)
        gain[chosen] = -1                                 # already taken
        top = int(np.argmax(gain))                        # ties -> first row = best score
        chosen.append(top)
        uncovered &= ~covers[top]

    chosen.sort()                                         # back to score order
    labels = [", ".join(p for p, hit in zip(picks, covers[i]) if hit) for i in chosen]
    return ranked.iloc[chosen].assign(covers=labels)


def recommend(user, restaurants, k=TOP_K, method="svm", cover=True):
    """Top-k restaurants for one user, best first.

    user: one entry of mock_users.json.
    restaurants: the restaurant_vectors table (default index).
    method: "svm" (default) or "cosine" (the existing score_restaurants baseline).
    cover: for people with 2+ tapped bubbles, pick the top-k so each bubble is
        represented where possible (see "Covering every pick" at the top).
    Returns a DataFrame: business_id, name, categories, stars, n_reviews, score
    (higher = better; its scale depends on the method), cosine (similarity to the
    user's likes vector, NaN for people with no preferences) and covers (which
    tapped bubbles the restaurant represents; empty when cover is off).
    """
    if method not in ("svm", "cosine"):
        raise ValueError(f"unknown method: {method}")

    profile = profile_from_user(user, restaurants)
    V = np.vstack(restaurants["vector"].values)
    likes = profile["likes"]
    picks = [*user.get("cravings", []), *user.get("specific", [])]
    P = pick_vectors(picks) if likes is not None and picks else None   # embedded once, used twice

    ok = allowed(profile, restaurants)                    # price range + must-have diets

    if likes is None:                                     # skipped: nothing to learn from
        score = (restaurants["stars"] * np.log1p(restaurants["n_reviews"])).values
        cosine = np.full(len(restaurants), np.nan)
    else:
        cosine = V @ likes
        base = score_restaurants(profile, restaurants)    # cosine baseline (likes - dislikes + diet bonus)
        score = (svm_scores(V, base, likes, profile["dislikes"], pick_vecs=P)
                 if method == "svm" else base)
        if method == "svm" and DIET_PREFER_BONUS and profile["diets_prefer"]:
            for diet in profile["diets_prefer"]:          # same substring rule as taste_profile
                tagged = restaurants["categories"].str.contains(DIET_CATEGORIES[diet], na=False)
                score = score + DIET_PREFER_BONUS * tagged.to_numpy()

    out = restaurants[["business_id", "name", "categories", "stars", "n_reviews"]].copy()
    out["score"] = score
    out["cosine"] = cosine
    out = out[ok].sort_values("score", ascending=False)
    out = out[~out["name"].str.lower().str.strip().duplicated()]   # one location per chain

    if cover and likes is not None and len(picks) >= 2:
        out = cover_top_k(out, V, picks, k, P)
    else:
        out = out.assign(covers="")
    return out.head(k).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Top restaurant matches for mock users.")
    parser.add_argument("--user", help="a user_id from mock_users.json, e.g. U0002")
    parser.add_argument("--limit", type=int, default=3,
                        help="how many users to show when --user is not given")
    parser.add_argument("--method", choices=["svm", "cosine"], default="svm")
    parser.add_argument("--no-cover", action="store_true",
                        help="skip the step that represents each tapped bubble")
    args = parser.parse_args()

    restaurants = pd.read_parquet(VECTORS)
    users = json.loads(USERS.read_text())
    if args.user:
        users = [u for u in users if u["user_id"] == args.user]
        if not users:
            raise SystemExit(f"No user {args.user} in {USERS}")
    else:
        users = users[:args.limit]

    for u in users:
        print(f"\n{u['user_id']}  likes: {u['loves_text'] or '(nothing)'}"
              f"  | avoids: {u['dislikes_text'] or '-'}  | loved: {u['loved_restaurants'] or '-'}")
        top = recommend(u, restaurants, method=args.method, cover=not args.no_cover)
        print(top[["name", "stars", "score", "cosine", "covers"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()