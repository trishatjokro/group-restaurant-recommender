import json
from pathlib import Path
import pandas as pd

from taste_profile import build_taste_profile
from group_recommender import recommend_group

ROOT = Path(__file__).resolve().parents[1]

restaurants = pd.read_parquet(
    ROOT / "data" / "output" / "restaurant_vectors.parquet"
)

with open(ROOT / "data" / "mock" / "mock_users.json") as f:
    mock_users = json.load(f)

mock_users = mock_users[:3]

print("Testing with", len(mock_users), "users")

# testing: group members with very different tastes
user1 = build_taste_profile(
    restaurants,
    loves_text = "pizza burgers american food cheesesteaks",
    dislikes_text = "sushi seafood japanese",
    prices = (1, 2)
)

user2 = build_taste_profile(
    restaurants,
    loves_text = "sushi japanese seafood ramen",
    dislikes_text = "burgers fast food cheesesteaks",
    prices = (1, 2)
)

user3 = build_taste_profile(
    restaurants,
    loves_text = "indian mediterranean vegetarian thai",
    dislikes_text = "pizza burgers fast food",
    prices = (1, 2)
)

taste_profiles = [user1, user2, user3]


results = recommend_group(
    taste_profiles,
    restaurants,
    top_n = 5
)

print("\nTop 5 by Average:")
print(
    results.sort_values("average_score", ascending=False)[
        ["name", "average_score"]
    ].head(5)
)

print("\nTop 5 by Least Misery:")
print(
    results.sort_values("least_misery_score", ascending=False)[
        ["name", "least_misery_score"]
    ].head(5)
)

print("\nTop 5 by Consensus:")
print(
    results.sort_values("consensus_score", ascending=False)[
        ["name", "consensus_score"]
    ].head(5)
)