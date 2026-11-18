import json
from pathlib import Path

import pandas as pd

from model.hybrid_model import hybrid_recommend


RESTAURANTS = Path("data") / "output" / "restaurant_vectors.parquet"
USERS = Path("data") / "mock" / "mock_users.json"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

restaurants = pd.read_parquet(RESTAURANTS)

with open(USERS) as f:
    users = json.load(f)


# ------------------------------------------------------------
# Pick a small test group
# ------------------------------------------------------------

group = users[:3]


# ------------------------------------------------------------
# Display users + preferences
# ------------------------------------------------------------

print("\nUsers in this group:")

for user in group:
    print(
        user["user_id"],
        "| cravings:", user.get("cravings", []),
        "| specific:", user.get("specific", []),
        "| diets:", user.get("diets", {})
    )


# ------------------------------------------------------------
# Run hybrid recommender
#
# alpha = 0.5 means:
# 50% SVM + 50% cosine
# ------------------------------------------------------------

results = hybrid_recommend(
    group,
    restaurants,
    k=5,
    alpha=0.5
)


# ------------------------------------------------------------
# Show main group results
# ------------------------------------------------------------

print("\nTop 5 recommendations:")

print(
    results[
        [
            "name",
            "average",
            "least_misery",
            "disagreement",
            "consensus",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Show each user's SVM, cosine, and hybrid score
# ------------------------------------------------------------

print("\nIndividual scores:")

columns = ["name"]

for user in group:
    user_id = user["user_id"]

    svm_col = f"{user_id}_svm"
    cosine_col = f"{user_id}_cosine"
    hybrid_col = f"{user_id}_hybrid"

    if svm_col in results.columns:
        columns.extend(
            [
                svm_col,
                cosine_col,
                hybrid_col,
            ]
        )

columns.append("consensus")

print(
    results[
        columns
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Verify that hybrid = 50% SVM + 50% cosine
# ------------------------------------------------------------

print("\nChecking hybrid score calculation:")

for user in group:
    user_id = user["user_id"]

    svm_col = f"{user_id}_svm"
    cosine_col = f"{user_id}_cosine"
    hybrid_col = f"{user_id}_hybrid"

    if svm_col not in results.columns:
        continue

    first = results.iloc[0]

    expected = (
        0.5 * first[svm_col]
        + 0.5 * first[cosine_col]
    )

    actual = first[hybrid_col]

    print(
        f"{user_id}: "
        f"expected hybrid = {expected:.6f}, "
        f"actual hybrid = {actual:.6f}"
    )