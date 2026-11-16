"""
Mock onboarding answers (taste profiles) for TESTING the app engine and
recommend(). These are made up, so they can't measure accuracy; use real Yelp
users for that.

Answers use the same bubbles as the onboarding screens. loves_text and
dislikes_text are ready to pass to build_taste_profile().
Users aren't grouped; make groups by sampling 2-5 users when needed.

Run from the repo's top folder:  python model/make_mock_profiles.py
Output: data/mock/mock_users.json
"""

import json
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_USERS = 1_000
VECTORS = Path("data") / "cities" / "philadelphia" / "restaurant_vectors.parquet"
OUT = Path("data") / "mock" / "mock_users.json"

CUISINES = ["French", "Japanese", "Thai", "Korean", "Indian", "American (New)", "American (Traditional)","Italian", "Mediterranean", "Chinese", "Vietnamese", "Mexican"]
FOODS = ["Sandwiches", "Pizza", "Breakfast & Brunch","Fast Food", "Burgers", "Coffee & Tea","Seafood", "Cafes", "Salad", "Delis","Chicken", "Cheesesteak"]
BUDGET=["$","$$","$$$"]
AVOID = ["Seafood", "Spicy", "Loud", "Long waits", "Fast food", "Chains", "Expensive"]
DIETS = ["vegetarian", "vegan", "halal", "gluten-free"]

rng = random.Random(SEED)


def user(user_id, cuisines=(), foods=(), avoid=(), prices=(), diets=None, skipped=False):
    return {
        "user_id": user_id,
        "skipped": skipped,
        "cuisines": list(cuisines),
        "foods": list(foods),
        "avoid": list(avoid),
        "prices": list(prices),
        "diets": diets or {},
    }


def random_user(user_id, popular):
    roll = rng.random()
    if roll < 0.10:                       # pressed Skip
        return user(user_id, skipped=True)
    if roll < 0.13:                       # answered nothing but didn't skip
        return user(user_id)
    if roll < 0.18:                       # only picked restaurants they love
        return user(user_id, loved=rng.sample(popular, rng.randint(1, 3)))
    diets = {}
    if rng.random() < 0.15:
        diets[rng.choice(DIETS)] = rng.choice(["must", "prefer"])
    return user(
        user_id,
        user_id,
        cuisines=rng.sample(CUISINES, rng.randint(0, 4)),
        foods=rng.sample(FOODS, rng.randint(0, 3)),
        avoid=rng.sample(AVOID, rng.randint(0, 2)),
        prices=sorted(rng.sample([1, 2, 3], rng.randint(0, 2))),
        diets=diets,
    )


def main():
    restaurants = pd.read_parquet(VECTORS, columns=["name", "n_reviews"])
    # Well-known places, so loved_restaurants look realistic and match the vectors file.
    popular = (restaurants.sort_values("n_reviews", ascending=False)["name"]
               .drop_duplicates().head(300).tolist())

    users = [random_user(f"U{i:04d}", popular) for i in range(1, N_USERS + 1)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(users, indent=2))
    skipped = sum(u["skipped"] for u in users)
    print(f"Saved {len(users):,} users ({skipped} skipped) -> {OUT}")


if __name__ == "__main__":
    main()
