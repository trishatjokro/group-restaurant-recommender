import json
import pandas as pd

from model.hybrid_model import hybrid_recommend

def main():

    # Load restaurant dataset
    restaurants = pd.read_parquet(
        "data/output/restaurant_vectors.parquet"
    )

    # Load mock users
    with open("data/mock/mock_users.json") as f:
        users = json.load(f)

    # Select specific users by their IDs
    selected_ids = ["U0003", "U0012", "U0019"]

    group = [
        user for user in users
        if user["user_id"] in selected_ids
    ]

    # Verify all selected users were found
    if len(group) != len(selected_ids):
        raise ValueError("One or more selected users were not found.")

    print("\nGroup Members:")
    for user in group:
        print(f"- {user['user_id']}")

    # Run hybrid recommender
    recommendations = hybrid_recommend(
        users=group,
        restaurants=restaurants,
        k=5
    )

    print("\nTop 5 Group Restaurant Recommendations:\n")

    if recommendations.empty:
        print("No restaurants satisfy everyone's constraints.")
        return

    # Include individual SVM scores when available
    svm_columns = [
        col for col in recommendations.columns
        if col.endswith("_svm")
    ]

    display_columns = (
        ["name", "stars", "n_reviews"]
        + svm_columns
        + [
            "average",
            "least_misery",
            "disagreement",
            "consensus"
        ]
    )

    print(
        recommendations[display_columns]
        .round(3)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()
