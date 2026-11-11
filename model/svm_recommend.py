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
"must have" diets, not recommending places they already named as loved, and one
location per chain.

People who skipped (nothing to learn from) get the most popular well-rated
restaurants that fit their filters, the same fallback group_vector.py uses.

Run from the repo's top folder:
  python -m model.svm_recommend --user U0002
  python -m model.svm_recommend --limit 5 --method cosine     # plain cosine baseline
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC

from model.taste_profile import allowed, build_taste_profile, score_restaurants

CITY = "philadelphia"
VECTORS = Path("data") / "cities" / CITY / "restaurant_vectors.parquet"
USERS = Path("data") / "mock" / "mock_users.json"

TOP_K = 5
SVM_C = 10.0              # margin softness; higher = fit the labels more tightly
PSEUDO_POSITIVES = 10     # best cosine-scored restaurants, used as extra positives
PSEUDO_NEGATIVES = 10     # restaurants closest to the dislikes vector, used as extra negatives
NEGATIVE_SAMPLES = 300    # random "everything else" negatives
NEGATIVE_SKIP = 0.20      # never sample negatives from the top 20% by cosine score
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


def svm_scores(V, base_score, likes, dislikes=None, seed=SEED):
    """One SVM score per restaurant (higher = better match).

    V: (n, d) unit-length restaurant vectors. likes / dislikes: unit-length vectors.
    base_score: the plain cosine score per restaurant (score_restaurants); it picks
    the pseudo-labels, so the SVM starts from the baseline and sharpens it.
    """
    n = len(V)
    order = np.argsort(-base_score)                       # best cosine matches first
    pos_idx = order[:min(PSEUDO_POSITIVES, n)]

    # Random negatives from everything except the top matches (those might be good
    # matches that just weren't picked as positives).
    pool = np.setdiff1d(order[int(NEGATIVE_SKIP * n):], pos_idx)
    rng = np.random.default_rng(seed)
    neg_idx = rng.choice(pool, size=min(NEGATIVE_SAMPLES, len(pool)), replace=False)

    rows, labels = [likes[None, :], V[pos_idx]], [1] * (1 + len(pos_idx))
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
    clf.fit(cosine_similarity(X), y)                      # cosine kernel
    return clf.decision_function(cosine_similarity(V, X))


def recommend(user, restaurants, k=TOP_K, method="svm", exclude_loved=True):
    """Top-k restaurants for one user, best first.

    user: one entry of mock_users.json.
    restaurants: the restaurant_vectors table (default index).
    method: "svm" (default) or "cosine" (the existing score_restaurants baseline).
    exclude_loved: don't recommend places the user already listed as loved.
    Returns a DataFrame: business_id, name, categories, stars, n_reviews, score
    (higher = better; its scale depends on the method) and cosine (similarity to
    the user's likes vector, NaN for people with no preferences).
    """
    if method not in ("svm", "cosine"):
        raise ValueError(f"unknown method: {method}")

    profile = profile_from_user(user, restaurants)
    V = np.vstack(restaurants["vector"].values)
    likes = profile["likes"]

    ok = allowed(profile, restaurants)                    # price range + must-have diets
    if exclude_loved:
        ok &= ~restaurants["business_id"].isin(profile["visited"]).values

    if likes is None:                                     # skipped: nothing to learn from
        score = (restaurants["stars"] * np.log1p(restaurants["n_reviews"])).values
        cosine = np.full(len(restaurants), np.nan)
    else:
        cosine = V @ likes
        base = score_restaurants(profile, restaurants)    # cosine baseline (likes - dislikes + diet bonus)
        score = svm_scores(V, base, likes, profile["dislikes"]) if method == "svm" else base

    out = restaurants[["business_id", "name", "categories", "stars", "n_reviews"]].copy()
    out["score"] = score
    out["cosine"] = cosine
    out = out[ok].sort_values("score", ascending=False)
    out = out[~out["name"].str.lower().str.strip().duplicated()]   # one location per chain
    return out.head(k).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Top restaurant matches for mock users.")
    parser.add_argument("--user", help="a user_id from mock_users.json, e.g. U0002")
    parser.add_argument("--limit", type=int, default=3,
                        help="how many users to show when --user is not given")
    parser.add_argument("--method", choices=["svm", "cosine"], default="svm")
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
        top = recommend(u, restaurants, method=args.method)
        print(top[["name", "stars", "score", "cosine"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
