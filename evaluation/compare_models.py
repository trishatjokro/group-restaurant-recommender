"""
Compare every group model on the same tests, with paired statistical tests.

Run from the repo's top folder (takes ~10 min, mostly training one SVM per person):
    python -m evaluation.compare_models
    python -m evaluation.compare_models --out some/other/folder

Models (each one gives every restaurant a score for the group; higher = better)
-------------------------------------------------------------------------------
  Average                  baseline: cosine to the average group vector (model/group_vector.py)
  Least misery             baseline: each restaurant gets the score of its least-happy member
  SVM (Anshika)            each person's SVM score (model/svm_recommend.py), rank-scaled to 0-1,
                           averaged over the group
  Consensus (Evelyn)       mean - std of members' cosine scores (model/group_recommender.py)
  Hybrid (SVM + consensus) mean - std of members' rank-scaled SVM scores (model/hybrid_model.py)
  Popularity (ref)         stars * log(1 + reviews), ignores taste: the "no taste model" yardstick
  Random (ref)             random scores: the floor
Members with no preferences are left out of the group score; if everyone skipped,
every model falls back to popularity (as the app does).

Test A: real Yelp friend pairs (the "guessing game")
----------------------------------------------------
Two Yelp users who reviewed many of the same restaurants (data/.../friend_pairs.parquet).
Find a restaurant BOTH liked, hide it, build each person's taste from their OTHER liked
restaurants, and let the model rank all restaurants (places either person already liked
are removed). Where does the hidden one land?
  hit10 / hit50   hidden restaurant is in the top 10 / top 50
  rank            its position (1 = top)
"Liked" = rated above the reviewer's own average (model/embed_restaurants.liked_reviews).

Test B: simulated groups of 2-10 people
---------------------------------------
Random groups of mock users (data/mock/mock_users.json), 60 per size per re-run.
Price and must-have diet filters apply to everyone. Each model returns its top 5
(one location per name). A member is "served" if at least one of the 5 has a Yelp
cuisine tag matching a bubble they tapped. The models never see those tags (the SVM
and cosine models only see text embeddings), so grading with them is not circular.
  served            share of members served
  nobody_left_out   every member served (MAIN METRIC, chosen before running)

Statistics
----------
Every model is scored on the SAME pairs and groups, so all tests are paired:
  McNemar (exact)        yes/no metrics: hit10, nobody_left_out
  Wilcoxon signed-rank   rank, served
  paired bootstrap       95% CI of the difference (2,000 resamples)
  Holm correction        over all comparisons in the table (28)
Reliability: 5 re-runs (seeds), each with fresh pairs and groups, plus results by group size.

Outputs (in --out, default results/model_comparison/):
  friend_pairs.csv   one row per pair x model
  mock_groups.csv    one row per group x model
  paired_tests.csv   every comparison: difference, 95% CI, p, Holm p
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from model.embed_restaurants import liked_reviews
from model.group_recommender import group_scores                      # Evelyn
from model.group_vector import group_vector                           # baseline: average
from model.hybrid_model import individual_svm_scores, normalize_scores  # Evelyn's hybrid
from model.svm_recommend import profile_from_user, svm_scores         # Anshika
from model.taste_profile import allowed, score_restaurants

CITY = Path("data") / "cities" / "philadelphia"
MOCK_USERS = Path("data") / "mock" / "mock_users.json"
SEEDS = [0, 1, 2, 3, 4]       # 5 re-runs
PAIRS_PER_SEED = 1000         # Test A: 5,000 friend pairs in total
GROUPS_PER_SIZE = 60          # Test B: per size per re-run -> 2,700 groups in total
SIZES = range(2, 11)
TOP_K = 5
BOOTSTRAP = 2000

BASELINES = ["Average", "Least misery"]
TEAM = ["SVM (Anshika)", "Consensus (Evelyn)", "Hybrid (SVM + consensus)"]
MODELS = BASELINES + TEAM + ["Popularity (ref)", "Random (ref)"]

# Yelp cuisine tags that count as "fits" each bubble (used ONLY for grading Test B)
TAGS = {"French": r"\bFrench\b", "Japanese": r"Japanese|Sushi|Ramen", "Thai": r"\bThai\b",
        "Korean": r"Korean", "Indian": r"Indian|Pakistani|Himalayan", "American (New)": r"American \(New\)",
        "Italian": r"Italian", "American (Traditional)": r"American \(Traditional\)",
        "Mediterranean": r"Mediterranean|Greek|Middle Eastern", "Chinese": r"Chinese|Szechuan|Cantonese|Dim Sum",
        "Vietnamese": r"Vietnamese", "Mexican": r"Mexican|Tacos", "Sandwiches": r"Sandwiches", "Pizza": r"Pizza",
        "Breakfast & Brunch": r"Breakfast & Brunch", "Fast Food": r"Fast Food", "Burgers": r"Burgers",
        "Coffee & Tea": r"Coffee & Tea|Cafes", "Seafood": r"Seafood", "Cafes": r"Cafes|Coffee & Tea",
        "Salad": r"Salad", "Delis": r"Delis", "Chicken": r"Chicken", "Cheesesteak": r"Cheesesteaks"}


def normalize(x):
    return x / np.linalg.norm(x)


def all_model_scores(profiles, restaurants, V, popularity, rng):
    """One score per restaurant for each model. Each profile with preferences
    carries "_svm": that person's rank-scaled SVM scores (computed once)."""
    active = [p for p in profiles if p["likes"] is not None]
    if not active:                                           # everyone skipped
        return {m: rng.random(len(V)) if m == "Random (ref)" else popularity for m in MODELS}
    cosine = np.vstack([score_restaurants(p, restaurants) for p in active])
    svm = np.vstack([p["_svm"] for p in active])
    return {
        "Average": V @ group_vector(profiles, restaurants)["vector"],
        "Least misery": cosine.min(axis=0),
        "SVM (Anshika)": svm.mean(axis=0),
        "Consensus (Evelyn)": group_scores(active, restaurants)[2],
        "Hybrid (SVM + consensus)": svm.mean(axis=0) - svm.std(axis=0),
        "Popularity (ref)": popularity,
        "Random (ref)": rng.random(len(V)),
    }


def test_friend_pairs(restaurants, V, popularity):
    """Test A: hide a restaurant both friends liked; where does each model rank it?"""
    idx = {b: i for i, b in enumerate(restaurants["business_id"])}
    reviews = pd.read_parquet(CITY / "reviews_clean.parquet", columns=["user_id", "business_id", "stars", "date"])
    liked = liked_reviews(reviews)
    liked = liked[liked.business_id.isin(idx)].sort_values("date").drop_duplicates(["user_id", "business_id"], keep="last")
    liked = liked.groupby("user_id").apply(lambda d: dict(zip(d.business_id, d.date)), include_groups=False).to_dict()

    def profile(known):                     # taste = mean of the restaurants they liked (minus the hidden one)
        p = {"likes": normalize(V[[idx[b] for b in known]].mean(axis=0)), "dislikes": None,
             "visited": set(known), "prices": set(), "diets_must": set(), "diets_prefer": set()}
        p["_svm"] = normalize_scores(svm_scores(V, score_restaurants(p, restaurants), p["likes"], None))
        return p

    pairs = []
    friends = pd.read_parquet(CITY / "friend_pairs.parquet").sample(frac=1, random_state=42)
    for a, b in zip(friends.user_a, friends.user_b):
        la, lb = liked.get(a, {}), liked.get(b, {})
        shared = set(la) & set(lb)
        if not shared:
            continue
        hidden = max(shared, key=lambda x: max(la[x], lb[x]))        # their most recent shared favourite
        ka, kb = [x for x in la if x != hidden], [x for x in lb if x != hidden]
        if len(ka) >= 2 and len(kb) >= 2:
            pairs.append((idx[hidden], ka, kb))
        if len(pairs) == PAIRS_PER_SEED * len(SEEDS):
            break

    rows = []
    for case, (hidden, ka, kb) in enumerate(pairs):
        rng = np.random.default_rng(case)
        already = restaurants["business_id"].isin(set(ka) | set(kb)).values
        for model, s in all_model_scores([profile(ka), profile(kb)], restaurants, V, popularity, rng).items():
            s = np.where(already, -np.inf, s)
            rank = int((s > s[hidden]).sum() + 1)
            rows.append(dict(case=case, seed=SEEDS[case % len(SEEDS)], model=model,
                             rank=rank, hit10=rank <= 10, hit50=rank <= 50))
    return pd.DataFrame(rows)


def test_mock_groups(restaurants, V, popularity):
    """Test B: does each member get a restaurant matching a cuisine they tapped?"""
    names = restaurants["name"].values
    fits = {b: restaurants["categories"].fillna("").str.contains(p).values for b, p in TAGS.items()}
    users = json.loads(MOCK_USERS.read_text())
    profiles = [profile_from_user(u, restaurants) for u in users]
    for u, p in zip(users, profiles):                        # SVM once per person, reused in every group
        if p["likes"] is not None:
            p["_svm"] = individual_svm_scores(u, restaurants, V, p)
    allowed_for = [allowed(p, restaurants) for p in profiles]
    user_fit = [np.any([fits[b] for b in u["cravings"] + u["specific"]], axis=0)
                if u["cravings"] + u["specific"] else None for u in users]

    def top(scores, ok):
        out, seen = [], set()
        for j in np.argsort(-np.where(ok, scores, -np.inf)):
            if not ok[j] or names[j] in seen:
                continue
            seen.add(names[j]); out.append(j)
            if len(out) == TOP_K:
                break
        return out

    rows, case = [], 0
    for seed in SEEDS:
        rng = np.random.default_rng(100 + seed)
        for size in SIZES:
            made = 0
            while made < GROUPS_PER_SIZE:
                members = rng.choice(len(users), size, replace=False)
                graded = [i for i in members if user_fit[i] is not None]   # people who tapped something
                if not graded:
                    continue
                ok = np.logical_and.reduce([allowed_for[i] for i in members])
                ok &= ~restaurants["business_id"].isin(set().union(*[profiles[i]["visited"] for i in members])).values
                group = [profiles[i] for i in members]
                for model, s in all_model_scores(group, restaurants, V, popularity, rng).items():
                    picks = top(s, ok)
                    served = [bool(user_fit[i][picks].any()) for i in graded]
                    rows.append(dict(case=case, seed=seed, size=size, model=model,
                                     served=np.mean(served), nobody_left_out=all(served)))
                case += 1
                made += 1
    return pd.DataFrame(rows)


def paired_tests(A, B):
    """Each teammate model vs each baseline (and baseline vs baseline), Holm-corrected."""
    boot = np.random.default_rng(7)

    def values(D, model, metric):
        return D[D.model == model].sort_values("case")[metric].astype(float).values

    rows = []
    comparisons = [("Average", "Least misery")] + [(t, b) for t in TEAM for b in BASELINES]
    for model, base in comparisons:
        for D, test, metric, kind in [(A, "friend_pairs", "hit10", "McNemar"), (A, "friend_pairs", "rank", "Wilcoxon"),
                                      (B, "mock_groups", "nobody_left_out", "McNemar"), (B, "mock_groups", "served", "Wilcoxon")]:
            x, y = values(D, model, metric), values(D, base, metric)
            d = x - y
            lo, hi = np.percentile(d[boot.integers(0, len(d), (BOOTSTRAP, len(d)))].mean(axis=1), [2.5, 97.5])
            if kind == "McNemar":
                better, worse = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
                p = binomtest(better, better + worse, 0.5).pvalue if better + worse else 1.0
            else:
                p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
            rows.append(dict(model=model, baseline=base, test=test, metric=metric, stat_test=kind,
                             diff=d.mean(), ci_low=lo, ci_high=hi, p=p))
    T = pd.DataFrame(rows)
    order, holm, running = np.argsort(T.p.values), np.empty(len(T)), 0.0
    for r, i in enumerate(order):                            # Holm step-down
        running = max(running, min(1.0, (len(T) - r) * T.p.values[i]))
        holm[i] = running
    T["p_holm"] = holm
    return T


def main():
    parser = argparse.ArgumentParser(description="Compare group models with paired statistical tests.")
    parser.add_argument("--out", default="results/model_comparison", help="folder for the CSV outputs")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    restaurants = pd.read_parquet(CITY / "restaurant_vectors.parquet")
    V = np.vstack(restaurants["vector"].values)
    popularity = (restaurants["stars"] * np.log1p(restaurants["n_reviews"])).values

    A = test_friend_pairs(restaurants, V, popularity)
    B = test_mock_groups(restaurants, V, popularity)
    T = paired_tests(A, B)
    A.to_csv(out / "friend_pairs.csv", index=False)
    B.to_csv(out / "mock_groups.csv", index=False)
    T.to_csv(out / "paired_tests.csv", index=False)

    print(f"Test A: {A.case.nunique()} real friend pairs | Test B: {B.case.nunique()} simulated groups\n")
    summary = pd.concat([A.groupby("model")[["hit10", "hit50"]].mean(),
                         B.groupby("model")[["served", "nobody_left_out"]].mean()], axis=1).loc[MODELS]
    print((100 * summary).round(1).to_string(), "\n")
    print("Paired tests (diff = model - baseline; Holm-corrected):")
    for _, r in T.iterrows():
        scale, unit = (1, " ranks") if r.metric == "rank" else (100, " pts")
        print(f"  {r.model:26s} vs {r.baseline:13s} {r.metric:16s} {r.stat_test:9s} "
              f"{scale * r['diff']:+7.1f}{unit} [{scale * r.ci_low:+.1f}, {scale * r.ci_high:+.1f}]  Holm p={r.p_holm:.2g}")
    print("\nNobody left out by re-run (seed):")
    print((100 * B.groupby(["seed", "model"]).nobody_left_out.mean().unstack()[BASELINES + TEAM]).round(1).to_string())
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    main()
