"""
Compare group recommendation models on the same tests.

Run from repo root:
    python -m evaluation.compare_models
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from model.embed_restaurants import liked_reviews
from model.group_recommender import group_scores
from model.group_vector import group_vector
from model.hybrid_model import (
    hybrid_scores,
    individual_svm_scores,
    normalize_scores,
)
from model.svm_recommend import profile_from_user, svm_scores
from model.taste_profile import allowed, score_restaurants


MOCK_USERS = Path("data") / "mock" / "mock_users.json"

SEEDS = [0, 1, 2, 3, 4]
PAIRS_PER_SEED = 1000
GROUPS_PER_SIZE = 60
SIZES = range(2, 11)
TOP_K = 5
BOOTSTRAP = 2000


BASELINES = [
    "Average",
    "Least misery"
]

TEAM = [
    "SVM (Anshika)",
    "Consensus (Evelyn)",
    "Hybrid (SVM + consensus)"
]

MODELS = (
    BASELINES
    + TEAM
    + [
        "Popularity (ref)",
        "Random (ref)"
    ]
)


# Test A cannot run the real hybrid model because
# Yelp friend pairs do not contain questionnaire users.
TEST_A_MODELS = [
    "Average",
    "Least misery",
    "SVM (Anshika)",
    "Consensus (Evelyn)",
    "Popularity (ref)",
    "Random (ref)"
]


TAGS = {
    "French": r"\bFrench\b",
    "Japanese": r"Japanese|Sushi|Ramen",
    "Thai": r"\bThai\b",
    "Korean": r"Korean",
    "Indian": r"Indian|Pakistani|Himalayan",
    "American (New)": r"American \(New\)",
    "Italian": r"Italian",
    "American (Traditional)": r"American \(Traditional\)",
    "Mediterranean": r"Mediterranean|Greek|Middle Eastern",
    "Chinese": r"Chinese|Szechuan|Cantonese|Dim Sum",
    "Vietnamese": r"Vietnamese",
    "Mexican": r"Mexican|Tacos",
    "Sandwiches": r"Sandwiches",
    "Pizza": r"Pizza",
    "Breakfast & Brunch": r"Breakfast & Brunch",
    "Fast Food": r"Fast Food",
    "Burgers": r"Burgers",
    "Coffee & Tea": r"Coffee & Tea|Cafes",
    "Seafood": r"Seafood",
    "Cafes": r"Cafes|Coffee & Tea",
    "Salad": r"Salad",
    "Delis": r"Delis",
    "Chicken": r"Chicken",
    "Cheesesteak": r"Cheesesteaks"
}


def normalize(x):
    return x / np.linalg.norm(x)


def all_model_scores(
    profiles,
    restaurants,
    V,
    popularity,
    rng
):
    """
    Scores models that only require taste profiles.

    Hybrid is handled separately in Test B.
    """

    active = [
        p for p in profiles
        if p["likes"] is not None
    ]

    if not active:
        return {
            "Average": popularity,
            "Least misery": popularity,
            "SVM (Anshika)": popularity,
            "Consensus (Evelyn)": popularity,
            "Popularity (ref)": popularity,
            "Random (ref)": rng.random(len(V)),
        }

    cosine = np.vstack([
        score_restaurants(
            p,
            restaurants
        )
        for p in active
    ])

    svm = np.vstack([
        p["_svm"]
        for p in active
    ])

    return {
        "Average":
            V @ group_vector(
                profiles,
                restaurants
            )["vector"],

        "Least misery":
            cosine.min(axis=0),

        "SVM (Anshika)":
            svm.mean(axis=0),

        "Consensus (Evelyn)":
            group_scores(
                active,
                restaurants
            )[2],

        "Popularity (ref)":
            popularity,

        "Random (ref)":
            rng.random(len(V)),
    }


def test_friend_pairs(
    restaurants,
    V,
    popularity
):
    """
    Test A:
    Hide a restaurant both friends liked
    and see where each model ranks it.
    """

    idx = {
        b: i
        for i, b
        in enumerate(
            restaurants["business_id"]
        )
    }

    reviews = pd.read_parquet(
        Path("data")
        / "output"
        / "reviews_clean.parquet",
        columns=[
            "user_id",
            "business_id",
            "stars",
            "date"
        ]
    )

    liked = liked_reviews(
        reviews
    )

    liked = (
        liked[
            liked.business_id.isin(idx)
        ]
        .sort_values("date")
        .drop_duplicates(
            ["user_id", "business_id"],
            keep="last"
        )
    )

    liked = (
        liked.groupby("user_id")
        .apply(
            lambda d:
                dict(
                    zip(
                        d.business_id,
                        d.date
                    )
                ),
            include_groups=False
        )
        .to_dict()
    )

    def profile(known):

        p = {
            "likes":
                normalize(
                    V[
                        [idx[b] for b in known]
                    ].mean(axis=0)
                ),

            "dislikes":
                None,

            "visited":
                set(known),

            "prices":
                set(),

            "diets_must":
                set(),

            "diets_prefer":
                set()
        }

        p["_svm"] = normalize_scores(
            svm_scores(
                V,
                score_restaurants(
                    p,
                    restaurants
                ),
                p["likes"],
                None
            )
        )

        return p

    pairs = []

    friends = (
        pd.read_parquet(
            Path("data")
            / "cities"
            / "philadelphia"
            / "friend_pairs.parquet"
        )
        .sample(
            frac=1,
            random_state=42
        )
    )

    for a, b in zip(
        friends.user_a,
        friends.user_b
    ):

        la = liked.get(a, {})
        lb = liked.get(b, {})

        shared = (
            set(la)
            & set(lb)
        )

        if not shared:
            continue

        hidden = max(
            shared,
            key=lambda x:
                max(
                    la[x],
                    lb[x]
                )
        )

        ka = [
            x
            for x in la
            if x != hidden
        ]

        kb = [
            x
            for x in lb
            if x != hidden
        ]

        if (
            len(ka) >= 2
            and len(kb) >= 2
        ):
            pairs.append(
                (
                    idx[hidden],
                    ka,
                    kb
                )
            )

        if (
            len(pairs)
            == PAIRS_PER_SEED
            * len(SEEDS)
        ):
            break

    rows = []

    for case, (
        hidden,
        ka,
        kb
    ) in enumerate(pairs):

        rng = np.random.default_rng(
            case
        )

        already = (
            restaurants[
                "business_id"
            ]
            .isin(
                set(ka)
                | set(kb)
            )
            .values
        )

        model_scores = all_model_scores(
            [
                profile(ka),
                profile(kb)
            ],
            restaurants,
            V,
            popularity,
            rng
        )

        for model, s in model_scores.items():

            s = np.where(
                already,
                -np.inf,
                s
            )

            rank = int(
                (
                    s > s[hidden]
                ).sum()
                + 1
            )

            rows.append(
                dict(
                    case=case,
                    seed=SEEDS[
                        case
                        % len(SEEDS)
                    ],
                    model=model,
                    rank=rank,
                    hit10=rank <= 10,
                    hit50=rank <= 50
                )
            )

    return pd.DataFrame(
        rows
    )


def test_mock_groups(
    restaurants,
    V,
    popularity
):
    """
    Test B:
    Does each member receive at least one
    restaurant matching a cuisine they chose?
    """

    names = (
        restaurants[
            "name"
        ].values
    )

    fits = {
        b:
            restaurants[
                "categories"
            ]
            .fillna("")
            .str.contains(p)
            .values

        for b, p
        in TAGS.items()
    }

    users = json.loads(
        MOCK_USERS.read_text()
    )

    profiles = [
        profile_from_user(
            u,
            restaurants
        )
        for u in users
    ]

    # Train each user's standalone SVM ONCE.
    for u, p in zip(
        users,
        profiles
    ):

        if p["likes"] is not None:

            p["_svm"] = (
                individual_svm_scores(
                    u,
                    restaurants,
                    V,
                    p
                )
            )

    allowed_for = [
        allowed(
            p,
            restaurants
        )
        for p in profiles
    ]

    user_fit = [
        np.any(
            [
                fits[b]
                for b
                in (
                    u["cravings"]
                    + u["specific"]
                )
            ],
            axis=0
        )
        if (
            u["cravings"]
            + u["specific"]
        )
        else None

        for u in users
    ]


    def top(scores, ok):

        out = []
        seen = set()

        for j in np.argsort(
            -np.where(
                ok,
                scores,
                -np.inf
            )
        ):

            if (
                not ok[j]
                or names[j] in seen
            ):
                continue

            seen.add(
                names[j]
            )

            out.append(j)

            if (
                len(out)
                == TOP_K
            ):
                break

        return out


    rows = []
    case = 0

    for seed in SEEDS:

        rng = (
            np.random
            .default_rng(
                100 + seed
            )
        )

        for size in SIZES:

            made = 0

            while (
                made
                < GROUPS_PER_SIZE
            ):

                members = rng.choice(
                    len(users),
                    size,
                    replace=False
                )

                graded = [
                    i
                    for i in members
                    if user_fit[i]
                    is not None
                ]

                if not graded:
                    continue

                ok = (
                    np.logical_and.reduce([
                        allowed_for[i]
                        for i in members
                    ])
                )

                ok &= ~restaurants[
                    "business_id"
                ].isin(
                    set().union(
                        *[
                            profiles[i][
                                "visited"
                            ]
                            for i in members
                        ]
                    )
                ).values

                group_profiles = [
                    profiles[i]
                    for i in members
                ]

                group_users = [
                    users[i]
                    for i in members
                ]

                # --------------------------------
                # Regular models
                # --------------------------------

                model_scores = (
                    all_model_scores(
                        group_profiles,
                        restaurants,
                        V,
                        popularity,
                        rng
                    )
                )

                # --------------------------------
                # Actual hybrid model
                #
                # 50% SVM + 50% cosine
                # followed by consensus
                # --------------------------------

                hybrid_results = (
                    hybrid_scores(
                        group_users,
                        restaurants,
                        alpha=0.5
                    )
                )

                model_scores[
                    "Hybrid (SVM + consensus)"
                ] = (
                    hybrid_results[
                        "consensus"
                    ]
                    .to_numpy()
                )

                # --------------------------------
                # Evaluate all models
                # --------------------------------

                for (
                    model,
                    s
                ) in model_scores.items():

                    picks = top(
                        s,
                        ok
                    )

                    served = [
                        bool(
                            user_fit[i][
                                picks
                            ].any()
                        )
                        for i in graded
                    ]

                    rows.append(
                        dict(
                            case=case,
                            seed=seed,
                            size=size,
                            model=model,
                            served=np.mean(
                                served
                            ),
                            nobody_left_out=all(
                                served
                            )
                        )
                    )

                case += 1
                made += 1

    return pd.DataFrame(
        rows
    )


def paired_tests(
    A,
    B
):
    """
    Paired statistical tests.

    Hybrid is only evaluated on Test B because
    Test A does not contain questionnaire users.
    """

    boot = (
        np.random
        .default_rng(7)
    )


    def values(
        D,
        model,
        metric
    ):

        return (
            D[
                D.model
                == model
            ]
            .sort_values(
                "case"
            )[
                metric
            ]
            .astype(float)
            .values
        )


    rows = []

    comparisons = [
        (
            "Average",
            "Least misery"
        )
    ] + [
        (
            t,
            b
        )
        for t in TEAM
        for b in BASELINES
    ]


    for model, base in comparisons:

        tests = []

        # Hybrid is not available in Test A.
        if (
            model
            != "Hybrid (SVM + consensus)"
        ):

            tests += [
                (
                    A,
                    "friend_pairs",
                    "hit10",
                    "McNemar"
                ),

                (
                    A,
                    "friend_pairs",
                    "rank",
                    "Wilcoxon"
                )
            ]

        tests += [
            (
                B,
                "mock_groups",
                "nobody_left_out",
                "McNemar"
            ),

            (
                B,
                "mock_groups",
                "served",
                "Wilcoxon"
            )
        ]


        for (
            D,
            test,
            metric,
            kind
        ) in tests:

            x = values(
                D,
                model,
                metric
            )

            y = values(
                D,
                base,
                metric
            )

            d = (
                x - y
            )

            lo, hi = np.percentile(
                d[
                    boot.integers(
                        0,
                        len(d),
                        (
                            BOOTSTRAP,
                            len(d)
                        )
                    )
                ]
                .mean(axis=1),
                [
                    2.5,
                    97.5
                ]
            )


            if (
                kind
                == "McNemar"
            ):

                better = int(
                    (
                        (x == 1)
                        & (y == 0)
                    ).sum()
                )

                worse = int(
                    (
                        (x == 0)
                        & (y == 1)
                    ).sum()
                )

                p = (
                    binomtest(
                        better,
                        better + worse,
                        0.5
                    ).pvalue
                    if (
                        better
                        + worse
                    )
                    else 1.0
                )

            else:

                p = (
                    wilcoxon(
                        x,
                        y
                    ).pvalue
                    if np.any(
                        d != 0
                    )
                    else 1.0
                )


            rows.append(
                dict(
                    model=model,
                    baseline=base,
                    test=test,
                    metric=metric,
                    stat_test=kind,
                    diff=d.mean(),
                    ci_low=lo,
                    ci_high=hi,
                    p=p
                )
            )


    T = pd.DataFrame(
        rows
    )


    order = np.argsort(
        T.p.values
    )

    holm = np.empty(
        len(T)
    )

    running = 0.0


    for r, i in enumerate(
        order
    ):

        running = max(
            running,
            min(
                1.0,
                (
                    len(T)
                    - r
                )
                * T.p.values[i]
            )
        )

        holm[i] = running


    T[
        "p_holm"
    ] = holm

    return T


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare group models "
            "with paired statistical tests."
        )
    )

    parser.add_argument(
        "--out",
        default=(
            "results/"
            "model_comparison"
        ),
        help=(
            "folder for CSV outputs"
        )
    )


    out = Path(
        parser.parse_args().out
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )


    restaurants = pd.read_parquet(
        Path("data")
        / "output"
        / "restaurant_vectors.parquet"
    )


    V = np.vstack(
        restaurants[
            "vector"
        ].values
    )


    popularity = (
        restaurants[
            "stars"
        ]
        * np.log1p(
            restaurants[
                "n_reviews"
            ]
        )
    ).values


    A = test_friend_pairs(
        restaurants,
        V,
        popularity
    )


    B = test_mock_groups(
        restaurants,
        V,
        popularity
    )


    T = paired_tests(
        A,
        B
    )


    A.to_csv(
        out
        / "friend_pairs.csv",
        index=False
    )


    B.to_csv(
        out
        / "mock_groups.csv",
        index=False
    )


    T.to_csv(
        out
        / "paired_tests.csv",
        index=False
    )


    print(
        f"Test A: "
        f"{A.case.nunique()} "
        f"real friend pairs "
        f"| Test B: "
        f"{B.case.nunique()} "
        f"simulated groups\n"
    )


    summary = pd.concat(
        [
            A.groupby(
                "model"
            )[
                [
                    "hit10",
                    "hit50"
                ]
            ].mean(),

            B.groupby(
                "model"
            )[
                [
                    "served",
                    "nobody_left_out"
                ]
            ].mean()
        ],
        axis=1
    ).reindex(
        MODELS
    )


    print(
        (
            100
            * summary
        )
        .round(1)
        .to_string(),
        "\n"
    )


    print(
        "Paired tests "
        "(diff = model - baseline; "
        "Holm-corrected):"
    )


    for _, r in T.iterrows():

        scale, unit = (
            (1, " ranks")
            if (
                r.metric
                == "rank"
            )
            else (
                100,
                " pts"
            )
        )

        print(
            f"  "
            f"{r.model:26s} "
            f"vs "
            f"{r.baseline:13s} "
            f"{r.metric:16s} "
            f"{r.stat_test:9s} "
            f"{scale * r['diff']:+7.1f}"
            f"{unit} "
            f"["
            f"{scale * r.ci_low:+.1f}, "
            f"{scale * r.ci_high:+.1f}"
            f"] "
            f"Holm p="
            f"{r.p_holm:.2g}"
        )


    print(
        "\nNobody left out "
        "by re-run (seed):"
    )


    print(
        (
            100
            * B.groupby(
                [
                    "seed",
                    "model"
                ]
            )
            .nobody_left_out
            .mean()
            .unstack()[
                BASELINES
                + TEAM
            ]
        )
        .round(1)
        .to_string()
    )


    print(
        f"\nSaved to "
        f"{out}/"
    )


if __name__ == "__main__":
    main()