# 1. Score restaurants using both SVM and cosine similarity for each active user.
# 2. Normalize both score types to comparable percentile ranks.
# 3. Combine SVM and cosine scores into one individual hybrid score.
# 4. Keep skipped users neutral during preference aggregation.
# 5. Pass hybrid individual scores into group_recommender.py.
# 6. If all users skip preferences, use popularity as a fallback.
# 7. Apply everyone's hard constraints.
# 8. Return either all restaurants or top K recommendations.

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

model_dir = Path(__file__).resolve().parent
if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from model import group_recommender
from model import svm_recommend
from model.taste_profile import score_restaurants, allowed


# Weight given to SVM when combining SVM + cosine.
# 0.5 = 50% SVM, 50% cosine.
ALPHA = 0.5


def normalize_scores(scores):
    """
    Convert scores to percentile ranks [0, 1].

    Higher values indicate stronger relative preference.
    These are not predicted probabilities.
    """

    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Restaurant scores must be finite.")

    if len(scores) == 0:
        return scores

    if np.ptp(scores) < 1e-12:
        return np.full(len(scores), 0.5)

    ranks = pd.Series(scores).rank(
        method="average"
    ).to_numpy()

    return (ranks - 1) / (len(scores) - 1)


def individual_svm_scores(user, restaurants, V, profile):
    """
    Generate normalized SVM scores for one active user.

    Returns:
        One normalized SVM score per restaurant.
    """

    likes = profile["likes"]
    dislikes = profile["dislikes"]

    if likes is None:
        raise ValueError(
            "Cannot calculate SVM scores without user preferences."
        )

    # Cosine-based scores used by Anshika's SVM
    # to generate pseudo-labels.
    base_scores = score_restaurants(
        profile,
        restaurants
    )

    picks = [
        *user.get("cravings", []),
        *user.get("specific", [])
    ]

    P = (
        svm_recommend.pick_vectors(picks)
        if picks else None
    )

    # Run Anshika's SVM.
    scores = svm_recommend.svm_scores(
        V,
        base_scores,
        likes,
        dislikes,
        pick_vecs=P
    )

    # Preserve preferred-diet bonuses.
    if (
        svm_recommend.DIET_PREFER_BONUS
        and profile["diets_prefer"]
    ):
        for diet in profile["diets_prefer"]:

            tagged = restaurants[
                "categories"
            ].str.contains(
                svm_recommend.DIET_CATEGORIES[diet],
                na=False
            )

            scores = (
                scores
                + svm_recommend.DIET_PREFER_BONUS
                * tagged.to_numpy()
            )

    return normalize_scores(scores)


def individual_hybrid_scores(
    user,
    restaurants,
    V,
    profile,
    alpha=ALPHA
):
    """
    Combine SVM and cosine scores for one user.

    alpha controls the balance:

        alpha = 0.5
            50% SVM
            50% cosine

        alpha = 0.75
            75% SVM
            25% cosine

        alpha = 0.25
            25% SVM
            75% cosine

    Returns:
        svm_scores
        cosine_scores
        hybrid_scores
    """

    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    # -----------------------------------------
    # SVM scores
    # -----------------------------------------

    svm_scores = individual_svm_scores(
        user,
        restaurants,
        V,
        profile
    )

    # -----------------------------------------
    # Cosine scores
    # -----------------------------------------

    cosine_scores = score_restaurants(
        profile,
        restaurants
    )

    # Put cosine scores on the same [0, 1]
    # percentile-rank scale as SVM scores.
    cosine_scores = normalize_scores(
        cosine_scores
    )

    # -----------------------------------------
    # Combine the two models
    # -----------------------------------------

    hybrid_scores = (
        alpha * svm_scores
        + (1 - alpha) * cosine_scores
    )

    return (
        svm_scores,
        cosine_scores,
        hybrid_scores
    )


def hybrid_scores(
    users,
    restaurants,
    alpha=ALPHA
):
    """
    Calculate hybrid scores for ALL restaurants.

    Hybrid pipeline:

        SVM score --------\
                           -> combined individual score
        cosine score -----/

                    ↓

        consensus aggregation

                    ↓

        final group ranking

    Users without taste preferences remain neutral.

    If everyone skips preferences, popularity is used
    as a fallback.

    Returns:
        DataFrame containing all restaurants and:
            - average
            - least_misery
            - disagreement
            - consensus
            - eligible
            - individual SVM scores
            - individual cosine scores
            - individual hybrid scores
    """

    if not users:
        raise ValueError(
            "Group must contain at least one user."
        )

    if restaurants.empty:
        raise ValueError(
            "Restaurant dataset cannot be empty."
        )

    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    restaurants = restaurants.reset_index(drop=True)

    V = np.vstack(
        restaurants["vector"].values
    )

    active_profiles = []
    active_user_ids = []

    all_svm_scores = []
    all_cosine_scores = []
    all_hybrid_scores = []

    eligible = np.ones(
        len(restaurants),
        dtype=bool
    )

    # -----------------------------------------
    # STEP 1: Process every group member
    # -----------------------------------------

    for user in users:

        profile = svm_recommend.profile_from_user(
            user,
            restaurants
        )

        # Hard constraints apply to every member,
        # including users without taste preferences.
        user_allowed = np.asarray(
            allowed(
                profile,
                restaurants
            ),
            dtype=bool
        )

        eligible &= user_allowed

        # Users without preferences do not affect
        # preference aggregation.
        if profile["likes"] is None:
            continue

        # -----------------------------------------
        # STEP 2: Calculate individual
        # SVM + cosine + hybrid scores
        # -----------------------------------------

        (
            svm_scores,
            cosine_scores,
            combined_scores
        ) = individual_hybrid_scores(
            user,
            restaurants,
            V,
            profile,
            alpha=alpha
        )

        active_profiles.append(
            profile
        )

        active_user_ids.append(
            user.get(
                "user_id",
                f"User{len(active_profiles)}"
            )
        )

        all_svm_scores.append(
            svm_scores
        )

        all_cosine_scores.append(
            cosine_scores
        )

        all_hybrid_scores.append(
            combined_scores
        )

    # -----------------------------------------
    # STEP 3: Everyone skipped preferences
    # -----------------------------------------

    if not active_profiles:

        popularity = (
            restaurants["stars"].to_numpy()
            * np.log1p(
                restaurants[
                    "n_reviews"
                ].to_numpy()
            )
        )

        popularity = normalize_scores(
            popularity
        )

        average = popularity.copy()
        least_misery = popularity.copy()

        disagreement = np.zeros(
            len(restaurants)
        )

        consensus = popularity.copy()

    else:

        # -----------------------------------------
        # STEP 4: Give the HYBRID individual
        # scores to group_recommender.py
        # -----------------------------------------

        score_lookup = {
            id(profile): scores
            for profile, scores in zip(
                active_profiles,
                all_hybrid_scores
            )
        }

        def hybrid_score_restaurants(
            profile,
            restaurants
        ):
            return score_lookup[id(profile)]

        # group_recommender.py now receives the
        # blended SVM + cosine score for each user.
        #
        # It then performs:
        #   average
        #   least misery
        #   consensus
        with patch.object(
            group_recommender,
            "score_restaurants",
            hybrid_score_restaurants
        ):

            (
                average,
                least_misery,
                consensus,
                active_eligible
            ) = group_recommender.group_scores(
                active_profiles,
                restaurants
            )

        # Combine constraints from active and
        # skipped users.
        eligible &= np.asarray(
            active_eligible,
            dtype=bool
        )

        # Convert lists into matrices:
        #
        # users x restaurants
        all_svm_scores = np.vstack(
            all_svm_scores
        )

        all_cosine_scores = np.vstack(
            all_cosine_scores
        )

        all_hybrid_scores = np.vstack(
            all_hybrid_scores
        )

        # Disagreement is measured using the
        # combined hybrid scores.
        disagreement = np.std(
            all_hybrid_scores,
            axis=0
        )

    # -----------------------------------------
    # STEP 5: Build result DataFrame
    # -----------------------------------------

    results = restaurants.copy()

    results["average"] = average
    results["least_misery"] = least_misery
    results["disagreement"] = disagreement
    results["consensus"] = consensus
    results["eligible"] = eligible

    # -----------------------------------------
    # STEP 6: Add individual scores
    # -----------------------------------------

    if active_profiles:

        for i, user_id in enumerate(
            active_user_ids
        ):

            results[
                f"{user_id}_svm"
            ] = all_svm_scores[i]

            results[
                f"{user_id}_cosine"
            ] = all_cosine_scores[i]

            results[
                f"{user_id}_hybrid"
            ] = all_hybrid_scores[i]

    return results


def hybrid_recommend(
    users,
    restaurants,
    k=5,
    alpha=ALPHA
):
    """
    Return the top K hybrid recommendations.

    Uses hybrid_scores() to score every restaurant,
    then:
        - removes ineligible restaurants
        - ranks by consensus
        - removes duplicate restaurant names
        - returns top K
    """

    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    # Score every restaurant.
    results = hybrid_scores(
        users,
        restaurants,
        alpha=alpha
    )

    # -----------------------------------------
    # STEP 1: Remove restaurants violating
    # hard constraints
    # -----------------------------------------

    results = results[
        results["eligible"]
    ].copy()

    # -----------------------------------------
    # STEP 2: Rank by consensus
    # -----------------------------------------

    results = results.sort_values(
        "consensus",
        ascending=False,
        kind="stable"
    )

    # -----------------------------------------
    # STEP 3: Keep one location per name
    # -----------------------------------------

    results = results[
        ~results["name"]
        .str.lower()
        .str.strip()
        .duplicated()
    ]

    # -----------------------------------------
    # STEP 4: Return top K
    # -----------------------------------------

    return (
        results
        .head(k)
        .reset_index(drop=True)
    )