"""
Measure the SVM recommender (precision / coverage / F1) and draw a correlation
matrix of its scores. Run from the repo's top folder:

  python -m model.evaluate_svm                    # 200 random mock users
  python -m model.evaluate_svm --users 500 --seed 1

What "good" means here (and what it does not)
---------------------------------------------
The mock users are made up (see make_mock_profiles.py), so there is no true list of
restaurants each one "really likes". The check is the same STRICT fit check as
EXPERIMENTS.md #10-12: a result fits a person only if its Yelp cuisine tag matches
one of the bubbles they tapped. The model never sees those tags (coverage uses the
embeddings only), so grading with them is not circular. For each user, with 5 results:

  precision   share of the 5 results that fit at least one of their tapped bubbles
  coverage    share of their tapped bubbles with at least one fitting result
  F1          harmonic mean of precision and coverage

This "F1" is NOT the usual retrieval F1. Standard recall@5 would ask "how many of
ALL fitting restaurants did we return", which is tiny by construction (thousands of
Italian places, 5 slots). Coverage is the version that makes sense for a top-5 list.
Compare methods against each other and against the random row; the absolute number
only means something next to those.

Beyond cuisine fit: the other preferences in mock_users.json
---------------------------------------------------------
Each is graded from Yelp data the model never sees (categories and attributes):
  avoid_rate         share of results that hit something the user avoids (lower is
                     better). Gradable: Seafood, Fast food, Chains (a name with 3+
                     locations), Expensive ($$$) (price 3+), Loud (NoiseLevel loud).
  diet_prefer_rate   share of results tagged with a diet the user "prefers"
  diet_must_ok       share tagged with every diet they "must" have (rule check: 1.0)
  price_ok           share not outside their price range (rule check: 1.0)
  loved_fit          share sharing a cuisine tag with a restaurant they said they love
  mean_stars         quality guardrail, so a method can't "win" by recommending junk
  clean_precision    share of results that fit a tapped bubble AND break no gradable
                     rule above. clean_f1 = harmonic mean of that and coverage.
The two "rule check" rows should be 1.0 for every method, because price and "must"
diets are hard filters; a lower number means a bug. The interesting rows are
avoid_rate, diet_prefer_rate and loved_fit, which only the model's scores affect.

Not gradable from this data (so NOT measured): "Long waits", "Spicy", the free-text
avoid_other items (mushrooms, cilantro, crowds, ...) and allergies. The script says
how many users each of these affects.

Still not measured: whether a person would actually enjoy a restaurant. For that, use
the real-user guessing game from experiments #4 and #8 (hide a restaurant a Yelp user
liked, check whether it shows up in their top 10).

Outputs: results/svm_eval.csv (per-method summary) and
results/svm_correlation.png / .csv (Spearman correlation of restaurant scores).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

from model.svm_recommend import (USERS, VECTORS, TOP_K, profile_from_user,   # noqa: E402
                                 pick_vectors, recommend, svm_scores)
from model.taste_profile import DIET_CATEGORIES, allowed, score_restaurants   # noqa: E402

RESULTS = Path("results")
N_BOOT = 1000

# Yelp category tags that count as a fit for each bubble (default: the bubble's own name).
YELP_TAGS = {
    "Cheesesteak": ["Cheesesteaks"],
    "Chicken": ["Chicken Shop", "Chicken Wings"],
}

METHODS = {                               # label -> (method, cover)
    "cosine": ("cosine", False),
    "cosine + coverage": ("cosine", True),
    "svm": ("svm", False),
    "svm + coverage": ("svm", True),
}


def fit_matrix(categories, picks):
    """(results x picks) bool: does each result's Yelp category list match each pick?"""
    tag_sets = [{c.strip() for c in str(cats).split(",")} for cats in categories]
    return np.array([[bool(set(YELP_TAGS.get(p, [p])) & tags) for p in picks]
                     for tags in tag_sets], dtype=bool).reshape(len(tag_sets), len(picks))


def score_one(top, picks):
    """precision, coverage and F1 for one user's list of results."""
    fits = fit_matrix(top["categories"], picks)
    precision = fits.any(axis=1).mean() if len(top) else 0.0
    coverage = fits.any(axis=0).mean()
    f1 = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return precision, coverage, f1


CHAIN_MIN = 3                              # a name on 3+ rows counts as a chain
GENERIC_TAGS = {"Restaurants", "Food"}     # say nothing about cuisine
LOUD = {"loud", "very_loud"}
AVOID_DETECTORS = {                        # normalized avoid label -> facts row -> bool Series
    "seafood": lambda f: f["cats"].str.contains("Seafood", regex=False),
    "fast food": lambda f: f["cats"].str.contains("Fast Food", regex=False),
    "chains": lambda f: f["chain"],
    "expensive": lambda f: f["price"] >= 3,
    "loud": lambda f: f["loud"],
}
NOT_GRADABLE = {"long waits", "spicy"}      # no data for these in the restaurant table


def restaurant_facts(restaurants):
    """Per-restaurant facts used only for GRADING (the model never sees them)."""
    cats = restaurants["categories"].fillna("").astype(str)
    attrs = restaurants["attributes"].fillna("").astype(str)
    names = restaurants["name"].str.lower().str.strip()
    facts = pd.DataFrame({
        "cats": cats.to_numpy(),
        "tags": [frozenset(t.strip() for t in c.split(",") if t.strip()) for c in cats],
        "price": pd.to_numeric(attrs.str.extract(r'"RestaurantsPriceRange2": "(\d)"')[0],
                               errors="coerce").to_numpy(),
        "loud": attrs.str.extract(r"NoiseLevel\W+u?'?(\w+)")[0].str.lower().isin(LOUD).to_numpy(),
        "chain": (names.map(names.value_counts()) >= CHAIN_MIN).to_numpy(),
    }, index=restaurants["business_id"].to_numpy())
    return facts


def _norm_avoid(label):
    return label.lower().replace("($$$)", "").strip()


def _mean(a):
    return float(np.mean(a)) if len(a) else np.nan


def score_user(top, user, facts, tags_by_name):
    """All metrics for one user's list of results (NaN = the preference doesn't apply)."""
    picks = [*user["cravings"], *user["specific"]]
    precision, coverage, f1 = score_one(top, picks)
    out = {"precision": precision, "coverage": coverage, "f1": f1}
    n = len(top)
    ft = facts.loc[top["business_id"]]
    fits_any = fit_matrix(top["categories"], picks).any(axis=1)

    # things to avoid (skip anything the user also tapped as a craving)
    picked = {p.lower() for p in picks}
    avoids = [a for a in map(_norm_avoid, user["avoid"])
              if a in AVOID_DETECTORS and a not in picked]
    hit = np.zeros(n, dtype=bool)
    for a in avoids:
        hit |= AVOID_DETECTORS[a](ft).fillna(False).to_numpy(bool)
    out["avoid_rate"] = _mean(hit) if avoids else np.nan

    # price range: unknown price is allowed (same rule as taste_profile.allowed)
    prices = set(user["prices"])
    price_ok = np.ones(n, dtype=bool)
    if prices:
        known = ft["price"].notna().to_numpy()
        price_ok = ~(known & ~ft["price"].isin(prices).to_numpy())
    out["price_ok"] = _mean(price_ok) if prices else np.nan

    # diets: same substring rule as taste_profile.allowed
    has = lambda d: ft["cats"].str.contains(DIET_CATEGORIES[d], regex=False).to_numpy(bool)
    must = [d for d, m in user["diets"].items() if m == "must"]
    prefer = [d for d, m in user["diets"].items() if m == "prefer"]
    must_ok = np.logical_and.reduce([has(d) for d in must]) if must else np.ones(n, dtype=bool)
    out["diet_must_ok"] = _mean(must_ok) if must else np.nan
    out["diet_prefer_rate"] = (_mean(np.logical_or.reduce([has(d) for d in prefer]))
                               if prefer else np.nan)

    # restaurants they said they love: do results share a cuisine tag with them?
    loved = set().union(*[tags_by_name.get(name.lower().strip(), set())
                          for name in user["loved_restaurants"]])
    out["loved_fit"] = (_mean([bool(t & loved) for t in ft["tags"]]) if loved else np.nan)

    out["mean_stars"] = _mean(top["stars"].to_numpy())
    clean = fits_any & ~hit & price_ok & must_ok
    out["clean_precision"] = _mean(clean) if n else 0.0
    cp = out["clean_precision"]
    out["clean_f1"] = 2 * cp * coverage / (cp + coverage) if cp + coverage else 0.0
    return out


def random_top(user, restaurants, k, rng):
    """Baseline: k random restaurants that pass the user's price / diet rules."""
    profile = profile_from_user(user, restaurants)
    ok = allowed(profile, restaurants)
    ok &= ~restaurants["business_id"].isin(profile["visited"]).values
    return restaurants[ok].sample(min(k, int(ok.sum())), random_state=int(rng.integers(1 << 31)))


def evaluate(users, restaurants, k=TOP_K, seed=0, methods=None, with_random=True, quiet=False):
    """One row per (user, method) with precision, coverage, f1.

    methods: {label: (method, cover)}; default is all of METHODS."""
    methods = METHODS if methods is None else methods
    rng = np.random.default_rng(seed)
    facts = restaurant_facts(restaurants)
    tags_by_name = {}                                     # lowercase name -> cuisine tags
    for name, tags in zip(restaurants["name"].str.lower().str.strip(), facts["tags"]):
        tags_by_name.setdefault(name, set()).update(tags - GENERIC_TAGS)
    rows = []
    for i, u in enumerate(users, 1):
        picks = [*u["cravings"], *u["specific"]]
        results = {"random": random_top(u, restaurants, k, rng)} if with_random else {}
        for label, (method, cover) in methods.items():
            results[label] = recommend(u, restaurants, k=k, method=method, cover=cover)
        for label, top in results.items():
            rows.append({"user_id": u["user_id"], "n_picks": len(picks), "method": label,
                         **score_user(top, u, facts, tags_by_name)})
        if not quiet and i % 25 == 0:
            print(f"  evaluated {i}/{len(users)} users")
    return pd.DataFrame(rows)


def paired_diff(per_user, a, b, metric="f1", seed=0):
    """Mean of (a - b) per user with a 95% bootstrap CI. The same users scored both
    methods, so this is tighter than comparing the two methods' own intervals."""
    x = per_user[per_user["method"] == a].set_index("user_id")[metric]
    y = per_user[per_user["method"] == b].set_index("user_id")[metric]
    d = (x - y.reindex(x.index)).to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi


def summarize(per_user, seed=0):
    """Mean of each metric per method with a 95% bootstrap CI over users."""
    rng = np.random.default_rng(seed)
    out = []
    for method, g in per_user.groupby("method", sort=False):
        row = {"method": method, "users": len(g)}
        for metric in ("precision", "coverage", "f1"):
            x = g[metric].to_numpy()
            means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            row[metric] = x.mean()
            row[f"{metric}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


PREFERENCE_METRICS = [                     # (column, what it means)
    ("clean_f1", "fits a bubble + breaks no rule, with coverage (higher)"),
    ("clean_precision", "results that fit AND break no rule (higher)"),
    ("avoid_rate", "results hitting something the user avoids (LOWER)"),
    ("diet_prefer_rate", "results tagged with a preferred diet (higher)"),
    ("loved_fit", "results sharing a cuisine with a loved restaurant (higher)"),
    ("mean_stars", "average Yelp stars of the results (guardrail)"),
    ("diet_must_ok", "rule check: must be 1.0"),
    ("price_ok", "rule check: must be 1.0"),
]


def summarize_preferences(per_user, seed=0):
    """One row per metric, one column per method: mean [95% CI] (n users it applies to)."""
    rng = np.random.default_rng(seed)
    rows = []
    for metric, meaning in PREFERENCE_METRICS:
        row = {"metric": metric}
        for method, g in per_user.groupby("method", sort=False):
            x = g[metric].dropna().to_numpy()
            if len(x) == 0:
                row[method] = "-"
                continue
            means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
            row[method] = f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}] n={len(x)}"
        row["meaning"] = meaning
        rows.append(row)
    return pd.DataFrame(rows)


def not_gradable(users):
    """How many users have preferences this data cannot grade."""
    return {
        "avoids 'Long waits' or 'Spicy'": sum(bool(NOT_GRADABLE & {_norm_avoid(a) for a in u["avoid"]})
                                              for u in users),
        "free-text avoid (mushrooms, cilantro, ...)": sum(bool(u["avoid_other"]) for u in users),
        "allergies": sum(bool(u["allergies"] or u["allergies_other"]) for u in users),
    }


def correlation(users, restaurants, results_dir):
    """Spearman correlation between restaurant scores, averaged over users.

    Columns: svm (SVM decision value), cosine_base (cosine baseline incl. dislikes),
    cosine_likes (plain cosine to the likes vector), stars, n_reviews.
    """
    V = np.vstack(restaurants["vector"].values)
    mats = []
    for u in users:
        profile = profile_from_user(u, restaurants)
        if profile["likes"] is None:
            continue
        base = score_restaurants(profile, restaurants)
        picks = [*u["cravings"], *u["specific"]]          # same inputs recommend() gives the SVM
        P = pick_vectors(picks) if picks else None
        df = pd.DataFrame({
            "svm": svm_scores(V, base, profile["likes"], profile["dislikes"], pick_vecs=P),
            "cosine_base": base,
            "cosine_likes": V @ profile["likes"],
            "stars": restaurants["stars"].to_numpy(),
            "n_reviews": restaurants["n_reviews"].to_numpy(),
        })
        mats.append(df.rank().corr().to_numpy())          # ranks -> Pearson = Spearman
    cols = ["svm", "cosine_base", "cosine_likes", "stars", "n_reviews"]
    corr = pd.DataFrame(np.mean(mats, axis=0), index=cols, columns=cols)

    results_dir.mkdir(exist_ok=True)
    corr.round(3).to_csv(results_dir / "svm_correlation.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Spearman correlation (mean over users)")
    ax.set_title(f"How the scores relate ({len(mats)} users)")
    fig.tight_layout()
    fig.savefig(results_dir / "svm_correlation.png", dpi=150)
    plt.close(fig)
    return corr


def main():
    parser = argparse.ArgumentParser(description="Evaluate the SVM recommender on mock users.")
    parser.add_argument("--users", type=int, default=200, help="how many mock users to evaluate")
    parser.add_argument("--corr-users", type=int, default=30, help="users for the correlation matrix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    restaurants = pd.read_parquet(VECTORS)
    users = [u for u in json.loads(USERS.read_text())
             if not u["skipped"] and (u["cravings"] or u["specific"])]
    rng = np.random.default_rng(args.seed)
    chosen = [users[i] for i in rng.choice(len(users), min(args.users, len(users)), replace=False)]
    print(f"{len(restaurants):,} restaurants; evaluating {len(chosen)} users "
          f"(skipped users and users with no tapped bubbles are left out)")

    per_user = evaluate(chosen, restaurants, seed=args.seed)
    summary = summarize(per_user, seed=args.seed)
    RESULTS.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "svm_eval.csv", index=False)
    print("\n" + summary.round(3).to_string(index=False))

    print("\nPaired F1 difference (same users; the interval excluding 0 means a real gap):")
    for a, b in [("svm", "cosine"), ("svm + coverage", "cosine + coverage"),
                 ("cosine + coverage", "cosine"), ("svm + coverage", "svm")]:
        d, lo, hi = paired_diff(per_user, a, b, seed=args.seed)
        print(f"  {a} minus {b}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\nOther preferences (each metric only counts users it applies to; n = how many):")
    prefs = summarize_preferences(per_user, seed=args.seed)
    prefs.to_csv(RESULTS / "svm_eval_preferences.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(prefs.set_index("metric").drop(columns="meaning").T.to_string())
    print("\nWhat the rows mean:")
    for metric, meaning in PREFERENCE_METRICS:
        print(f"  {metric:<17}{meaning}")
    print("\nPreferences this data cannot grade (NOT measured), out of "
          f"{len(chosen)} evaluated users:")
    for what, count in not_gradable(chosen).items():
        print(f"  {count:>4} users: {what}")

    multi = per_user[per_user["n_picks"] >= 2]
    print("\nUsers with 2+ tapped bubbles only (where coverage can matter):")
    print(summarize(multi, seed=args.seed).round(3).to_string(index=False))

    print("\nSpearman correlation of scores (mean over users):")
    corr = correlation(chosen[:args.corr_users], restaurants, RESULTS)
    print(corr.round(2).to_string())
    print(f"\nSaved {RESULTS}/svm_eval.csv, svm_correlation.csv, svm_correlation.png")


if __name__ == "__main__":
    main()