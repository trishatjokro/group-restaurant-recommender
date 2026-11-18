# 1. Score restaurants using Anshika's SVM for active users.
# 2. Normalize individual SVM scores to comparable percentile ranks.
# 3. Keep skipped users neutral during preference aggregation.
# 4. Pass SVM scores into the original group_recommender.py.
# 5. If all users skip preferences, use popularity as a fallback.
# 6. Apply everyone's hard constraints.
# 7. Return top K restaurants with individual and group scores.

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

# Allow original group_recommender.py imports to work
model_dir = Path(__file__).resolve().parent
if str(model_dir) not in sys.path:
    sys.path.insert(0, str(model_dir))

from model import group_recommender
from model import svm_recommend
from model.taste_profile import score_restaurants, allowed


def normalize_scores(scores):
    """
    Convert individual scores to percentile ranks [0, 1].

    Higher values indicate stronger relative preference.
    These are not predicted probabilities.
    """
    scores = np.asarray(scores, dtype=float)

    if not np.isfinite(scores).all():
        raise ValueError("Restaurant scores must be finite.")

    if len(scores) == 0:
        return scores

    # All restaurants have the same score
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

    # Original cosine scores generate SVM pseudo-labels
    base_scores = score_restaurants(
        profile, restaurants
    )

    # Preserve individual selected preference bubbles
    picks = [
        *user.get("cravings", []),
        *user.get("specific", [])
    ]

    P = (
        svm_recommend.pick_vectors(picks)
        if picks else None
    )

    # Run Anshika's original SVM
    # cover_top_k() is never called.
    scores = svm_recommend.svm_scores(
        V,
        base_scores,
        likes,
        dislikes,
        pick_vecs=P
    )

    # Preserve optional preferred-diet bonuses
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

    # Normalize before consensus aggregation
    return normalize_scores(scores)


def hybrid_scores(users, restaurants):
    """
    Steps 1-5 of the hybrid model: score EVERY restaurant for the group.

    Used by hybrid_recommend() below and by the app's API (api/app.py),
    so both rank restaurants the same way.

    Returns a dict (one value per restaurant, in restaurants' row order):
        restaurants      the table with a fresh 0..n-1 index
        eligible         True where everyone's hard constraints are met
        average, least_misery, disagreement, consensus
        active_user_ids  users who gave preferences (skipped users are neutral)
        all_scores       (active users x restaurants) normalized SVM scores,
                         or an empty list if everyone skipped
    """

    if not users:
        raise ValueError("Group must contain at least one user.")

    if restaurants.empty:
        raise ValueError("Restaurant dataset cannot be empty.")

    # Reset index to align restaurant rows with embeddings
    restaurants = restaurants.reset_index(drop=True)

    # Restaurant embeddings
    V = np.vstack(restaurants["vector"].values)

    active_profiles = []
    active_user_ids = []
    all_scores = []

    # All restaurants initially eligible
    eligible = np.ones(len(restaurants), dtype=bool)

    # STEP 1: Process every group member
    for user in users:

        profile = svm_recommend.profile_from_user(
            user, restaurants
        )

        # Apply hard constraints for every group member,
        # including those who skipped preferences.
        user_allowed = np.asarray(
            allowed(profile, restaurants),
            dtype=bool
        )

        eligible &= user_allowed

        # STEP 2: Skip users without taste preferences
        if profile["likes"] is None:
            print(
                f"{user.get('user_id', 'Unknown user')}: "
                "No preferences, excluded from consensus."
            )
            continue

        # STEP 3: Calculate individual normalized SVM scores
        scores = individual_svm_scores(
            user,
            restaurants,
            V,
            profile
        )

        active_profiles.append(profile)
        active_user_ids.append(
            user.get("user_id", f"User{len(active_profiles)}")
        )
        all_scores.append(scores)

    # STEP 4: Determine whether to use consensus or popularity
    if not active_profiles:

        print(
            "\nAll users skipped preferences. "
            "Using popularity fallback."
        )

        # Same popularity formula as Anshika's SVM fallback
        popularity = (
            restaurants["stars"].to_numpy()
            * np.log1p(
                restaurants["n_reviews"].to_numpy()
            )
        )

        popularity = normalize_scores(popularity)

        # No individual SVM scores in the fallback case
        average = popularity.copy()
        least_misery = popularity.copy()
        disagreement = np.zeros(len(restaurants))
        consensus = popularity.copy()

    else:

        print(
            f"\nUsing SVM scores from "
            f"{len(active_profiles)} of {len(users)} users."
        )

        # STEP 5: Provide SVM scores to the original
        # group_recommender.py without editing it.

        score_lookup = {
            id(profile): scores
            for profile, scores in zip(
                active_profiles, all_scores
            )
        }

        def hybrid_score_restaurants(profile, restaurants):
            return score_lookup[id(profile)]

        # Temporarily replace individual scoring
        # while running your original consensus model.
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

        # Combine constraints from active and skipped users
        eligible &= np.asarray(
            active_eligible,
            dtype=bool
        )

        # Calculate disagreement among active users only
        all_scores = np.vstack(all_scores)
        disagreement = np.std(all_scores, axis=0)

    return {
        "restaurants": restaurants,
        "eligible": eligible,
        "average": average,
        "least_misery": least_misery,
        "disagreement": disagreement,
        "consensus": consensus,
        "active_user_ids": active_user_ids,
        "all_scores": all_scores,
    }


def hybrid_recommend(users, restaurants, k=5):
    """
    Generate group recommendations using SVM + Consensus.

    Users without taste preferences remain neutral.
    If everyone skips preferences, use popularity.

    Returns a DataFrame containing:
        - Restaurant information
        - Individual normalized SVM scores
        - Average group preference
        - Least misery
        - Disagreement
        - Consensus
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    # STEPS 1-5: score every restaurant for the group
    s = hybrid_scores(users, restaurants)
    restaurants = s["restaurants"]
    eligible = s["eligible"]
    average = s["average"]
    least_misery = s["least_misery"]
    disagreement = s["disagreement"]
    consensus = s["consensus"]
    active_user_ids = s["active_user_ids"]
    all_scores = s["all_scores"]
    active_profiles = active_user_ids  # only used to check "anyone active?"

    # STEP 6: Keep only restaurants satisfying all constraints
    results = restaurants.loc[
        eligible,
        [
            "business_id",
            "name",
            "categories",
            "stars",
            "n_reviews"
        ]
    ].copy()

    # STEP 7: Include group aggregation scores
    results["average"] = average[eligible]
    results["least_misery"] = least_misery[eligible]
    results["disagreement"] = disagreement[eligible]
    results["consensus"] = consensus[eligible]

    # STEP 8: Display each active user's normalized SVM score
    if active_profiles:
        for i, user_id in enumerate(active_user_ids):
            results[f"{user_id}_svm"] = all_scores[i, eligible]

    # STEP 9: Rank by highest consensus
    results = results.sort_values(
        "consensus",
        ascending=False,
        kind="stable"
    )

    # Keep only one location per restaurant chain/name
    results = results[
        ~results["name"]
        .str.lower()
        .str.strip()
        .duplicated()
    ]

    # STEP 10: Return top K recommendations
    return results.head(k).reset_index(drop=True)
