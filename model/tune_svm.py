"""
Try different SVM / coverage settings and report which one raises F1, WITHOUT
grading on the users it was tuned on.

  python -m model.tune_svm                       # 150 tuning users, 300 holdout users
  python -m model.tune_svm --tune-users 100 --holdout-users 200 --seed 3

How it stays honest
-------------------
1. The mock users are split into two disjoint groups: TUNE and HOLDOUT.
2. Every setting below is scored on the TUNE users only (F1 of "svm + coverage";
   F1 is defined in evaluate_svm.py).
3. The best setting is then scored ONCE on the HOLDOUT users, next to the current
   defaults and next to "cosine + coverage", with paired confidence intervals.
Report the HOLDOUT numbers. If the best setting is not clearly above the defaults
there, the tuning gain was luck; keep the defaults.

Takes a few minutes: each setting trains one SVM per user.
What each setting means is described next to the constants in svm_recommend.py.
"""

import argparse
import json

import numpy as np
import pandas as pd

import model.svm_recommend as sr
from model.evaluate_svm import evaluate, paired_diff, summarize, summarize_preferences

# Each entry changes the CURRENT defaults for that run only. Settings are grouped by
# what they test; "defaults" is the baseline everything is compared with.
GRID = {
    "defaults": {},
    # a smoother or sharper margin
    "C=1": {"SVM_C": 1.0},
    "C=100": {"SVM_C": 100.0},
    # more or fewer pseudo-labelled examples
    "positives=25": {"PSEUDO_POSITIVES": 25},
    "negatives=800": {"NEGATIVE_SAMPLES": 800},
    # local (rbf) kernel: it can circle two separate cuisine clusters, cosine cannot
    "rbf g=5": {"KERNEL": "rbf", "RBF_GAMMA": 5.0},
    "rbf g=10": {"KERNEL": "rbf", "RBF_GAMMA": 10.0},
    "rbf g=20": {"KERNEL": "rbf", "RBF_GAMMA": 20.0},
    # per-bubble positives: describe each tapped bubble as its own cluster
    "per-bubble 10": {"PER_PICK_POSITIVES": 10},
    "per-bubble 10 + rbf g=10": {"PER_PICK_POSITIVES": 10, "KERNEL": "rbf", "RBF_GAMMA": 10.0},
    "per-bubble 20 + rbf g=10": {"PER_PICK_POSITIVES": 20, "KERNEL": "rbf", "RBF_GAMMA": 10.0},
    "per-bubble 20 + rbf g=20": {"PER_PICK_POSITIVES": 20, "KERNEL": "rbf", "RBF_GAMMA": 20.0},
    # how the coverage step chooses (applies to cosine + coverage too)
    "cover top=10": {"COVER_TOP": 10},
    "cover top=30": {"COVER_TOP": 30},
    "cover z=1": {"COVER_Z": 1.0},
    "cover frac=0.3": {"COVER_SCORE_FRAC": 0.3},
    "cover frac=0.7": {"COVER_SCORE_FRAC": 0.7},
}
SVM_COVER = {"svm + coverage": ("svm", True)}
BOTH = {"svm + coverage": ("svm", True), "cosine + coverage": ("cosine", True)}


class Settings:
    """Temporarily set constants in svm_recommend, then restore them."""

    def __init__(self, changes):
        self.changes, self.saved = changes, {}

    def __enter__(self):
        for name, value in self.changes.items():
            self.saved[name] = getattr(sr, name)
            setattr(sr, name, value)

    def __exit__(self, *exc):
        for name, value in self.saved.items():
            setattr(sr, name, value)


def f1_of(users, restaurants, changes, seed, objective="f1"):
    with Settings(changes):
        per = evaluate(users, restaurants, seed=seed, methods=SVM_COVER,
                       with_random=False, quiet=True)
    return per[objective].mean()


DIET_BONUSES = [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]


def diet_bonus_sweep(restaurants, users, seed):
    """Try DIET_PREFER_BONUS values on every user who has a "prefer" diet.

    Only ~8% of users have one, so these users are too few for the main tuning run
    (their effect on overall F1 is tiny). Here we look at them directly: does the
    bonus raise how often their preferred diet shows up (diet_prefer_rate), and what
    does it cost in cuisine fit (f1) and rule-following (clean_f1)? The same users are
    used to choose and to report, so treat it as a sensible-value finder, not a test.
    """
    diet_users = [u for u in users if "prefer" in u["diets"].values()]
    print(f"{len(diet_users)} users have a diet they 'prefer' (the bonus only affects them)\n")
    rows = []
    for bonus in DIET_BONUSES:
        with Settings({"DIET_PREFER_BONUS": bonus}):
            per = evaluate(diet_users, restaurants, seed=seed, methods=SVM_COVER,
                           with_random=False, quiet=True)
        rows.append({"setting": f"svm + coverage, bonus={bonus}", **per[
            ["diet_prefer_rate", "f1", "clean_f1", "avoid_rate", "mean_stars"]].mean().to_dict()})
    ref = evaluate(diet_users, restaurants, seed=seed, methods={"cosine + coverage": ("cosine", True)},
                   with_random=False, quiet=True)
    rows.append({"setting": "cosine + coverage (reference)", **ref[
        ["diet_prefer_rate", "f1", "clean_f1", "avoid_rate", "mean_stars"]].mean().to_dict()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nPick the smallest bonus where diet_prefer_rate is close to the cosine row "
          "without f1 / clean_f1 dropping noticeably, then set DIET_PREFER_BONUS in "
          "model/svm_recommend.py.")


def main():
    parser = argparse.ArgumentParser(description="Tune SVM settings with a holdout check.")
    parser.add_argument("--diet-bonus", action="store_true",
                        help="only sweep DIET_PREFER_BONUS on users with a 'prefer' diet")
    parser.add_argument("--tune-users", type=int, default=150)
    parser.add_argument("--holdout-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", choices=["f1", "clean_f1"], default="f1",
                        help="f1 = cuisine fit only; clean_f1 = cuisine fit AND no avoided/"
                             "out-of-range results (see evaluate_svm.py)")
    args = parser.parse_args()

    restaurants = pd.read_parquet(sr.VECTORS)
    users = [u for u in json.loads(sr.USERS.read_text())
             if not u["skipped"] and (u["cravings"] or u["specific"])]
    if args.diet_bonus:
        diet_bonus_sweep(restaurants, users, args.seed)
        return
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(users))
    n_tune = min(args.tune_users, len(users) // 3)
    n_hold = min(args.holdout_users, len(users) - n_tune)
    tune = [users[i] for i in order[:n_tune]]
    holdout = [users[i] for i in order[n_tune:n_tune + n_hold]]
    print(f"{len(restaurants):,} restaurants | {len(tune)} tuning users, "
          f"{len(holdout)} holdout users (no overlap)\n")

    # Stage 1: score every setting on the tuning users.
    scores = {}
    for name, changes in GRID.items():
        scores[name] = f1_of(tune, restaurants, changes, args.seed, args.objective)
        print(f"  {name:<28} tune {args.objective} {scores[name]:.3f}")
    best = max(scores, key=scores.get)
    print(f"\nBest on tuning users: {best} ({scores[best]:.3f}; "
          f"defaults {scores['defaults']:.3f})")

    # Stage 2: one honest look at the holdout users.
    per = []
    for label in dict.fromkeys(["defaults", best]):       # just "defaults" if it won
        changes = GRID[label]
        with Settings(changes):
            p = evaluate(holdout, restaurants, seed=args.seed, methods=BOTH,
                         with_random=False, quiet=True)
        p["method"] = p["method"].map(lambda m: f"{m} [{label}]")
        per.append(p)
    per = pd.concat(per)
    # cosine + coverage is the same under both runs unless a coverage setting changed
    print("\nHOLDOUT results (users the settings never saw):")
    print(summarize(per).round(3).to_string(index=False))
    print("\nOther preferences on the same holdout users:")
    prefs = summarize_preferences(per, seed=args.seed)
    print(prefs.set_index("metric").drop(columns="meaning").T.to_string())

    a, b = f"svm + coverage [{best}]", "svm + coverage [defaults]"
    c = "cosine + coverage [defaults]"
    for x, y in (((a, b), (a, c)) if best != "defaults" else ((a, c),)):
        d, lo, hi = paired_diff(per, x, y, metric=args.objective, seed=args.seed)
        print(f"\n{x}\n  minus {y}: {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print(f"\nTo use '{best}', set these in model/svm_recommend.py: {GRID[best] or '(nothing: defaults won)'}")


if __name__ == "__main__":
    main()