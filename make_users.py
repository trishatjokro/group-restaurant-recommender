"""
Reads the city's cleaned reviews and the raw Yelp user file, and saves:
    data/cities/<city>/users.parquet         one row per reviewer in the city
    data/cities/<city>/friend_pairs.parquet  Yelp friends who both reviewed here,
                                             + how many restaurants they share
"""

import json
from pathlib import Path

import pandas as pd

# This is for my personal path. Change it to fit your own.
RAW = Path.home() / "Documents" / "Data SC" / "Yelp JSON" / "yelp_dataset"
CITY = "philadelphia"
FOLDER = Path("data") / "cities" / CITY


def main():
    reviews = pd.read_parquet(FOLDER / "reviews_clean.parquet")
    city_users = set(reviews["user_id"])

    # The user file is 3.4 GB, mostly friend lists, so read it line by line
    # and keep only this city's reviewers and their friends who are also here.
    rows = []
    with open(RAW / "yelp_academic_dataset_user.json", "rb") as f:
        for line in f:
            d = json.loads(line)
            if d["user_id"] in city_users:
                friends = [x for x in d["friends"].split(", ") if x in city_users]
                rows.append((d["user_id"], d["review_count"], d["yelping_since"], friends))
    users = pd.DataFrame(rows, columns=["user_id", "yelp_review_count",
                                        "yelping_since", "friends_in_city"])
    users["n_city_reviews"] = users["user_id"].map(reviews["user_id"].value_counts())

    # Friend pairs (each pair once) + restaurants both of them reviewed.
    visited = reviews.groupby("user_id")["business_id"].apply(set).to_dict()
    pairs = []
    for user, friends in zip(users["user_id"], users["friends_in_city"]):
        for friend in friends:
            if user < friend:
                pairs.append((user, friend, len(visited[user] & visited[friend])))
    friend_pairs = pd.DataFrame(pairs, columns=["user_a", "user_b", "n_shared_restaurants"])

    users.to_parquet(FOLDER / "users.parquet", index=False)
    friend_pairs.to_parquet(FOLDER / "friend_pairs.parquet", index=False)
    print(f"{len(users):,} users, {len(friend_pairs):,} friend pairs")
    print(f"pairs sharing 5+ restaurants: {(friend_pairs['n_shared_restaurants'] >= 5).sum():,}")


if __name__ == "__main__":
    main()
