"""
Mock onboarding answers (taste profiles) for TESTING the app engine and
recommend(). These are made up, so they can't measure accuracy; use real Yelp
users for that.

Answers follow the agreed questionnaire: dietary needs (must/prefer), food
allergies, cravings (cuisine), anything specific (food type), budget, things to
avoid, and restaurants they love. loves_text and dislikes_text are ready to pass
to build_taste_profile(). Allergies are only recorded: the data has no ingredient
info, so the app shows a "may contain ..." disclaimer instead of filtering.
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

DIETS = ["vegetarian", "vegan", "gluten-free", "halal"]
ALLERGIES = ["peanuts", "dairy", "shellfish", "wheat", "soy"]
ALLERGIES_OTHER = ["sesame", "tree nuts", "eggs", "mustard"]
CRAVINGS = ["French", "Japanese", "Thai", "Korean", "Indian", "American (New)", "Italian",
            "American (Traditional)", "Mediterranean", "Chinese", "Vietnamese", "Mexican"]
SPECIFIC = ["Sandwiches", "Pizza", "Breakfast & Brunch", "Fast Food", "Burgers", "Coffee & Tea",
            "Seafood", "Cafes", "Salad", "Delis", "Chicken", "Cheesesteak"]
AVOID = ["Seafood", "Spicy", "Loud", "Long waits", "Fast food", "Chains", "Expensive ($$$)"]
AVOID_OTHER = ["mushrooms", "cilantro", "crowds", "olives", "raw onions"]

rng = random.Random(SEED)


def bubble_text(labels):
    return ", ".join(b.replace(" ($$$)", "").lower() for b in labels)


def user(user_id, diets=None, allergies=(), allergies_other="", cravings=(), specific=(),
         prices=(), avoid=(), avoid_other="", loved=(), skipped=False):
    return {
        "user_id": user_id,
        "skipped": skipped,
        "diets": diets or {},
        "allergies": list(allergies),
        "allergies_other": allergies_other,
        "cravings": list(cravings),
        "specific": list(specific),
        "prices": list(prices),
        "avoid": list(avoid),
        "avoid_other": avoid_other,
        "loved_restaurants": list(loved),
        "loves_text": bubble_text([*cravings, *specific]),
        "dislikes_text": ", ".join(t for t in [bubble_text(avoid), avoid_other] if t),
    }


def random_user(user_id, popular):
    roll = rng.random()
    if roll < 0.10:                       # pressed Skip
        return user(user_id, skipped=True)
    if roll < 0.13:                       # answered nothing but didn't skip
        return user(user_id)
    if roll < 0.18:                       # only entered restaurants they love
        return user(user_id, loved=rng.sample(popular, rng.randint(1, 3)))
    diets = {}
    if rng.random() < 0.15:
        diets[rng.choice(DIETS)] = rng.choice(["must", "prefer"])
    allergies = rng.sample(ALLERGIES, rng.randint(1, 2)) if rng.random() < 0.15 else []
    return user(
        user_id,
        diets=diets,
        allergies=allergies,
        allergies_other=rng.choice(ALLERGIES_OTHER) if rng.random() < 0.05 else "",
        cravings=rng.sample(CRAVINGS, rng.randint(1, 3)),
        specific=rng.sample(SPECIFIC, rng.randint(0, 3)),
        prices=sorted(rng.sample([1, 2, 3], rng.randint(0, 2))),
        avoid=rng.sample(AVOID, rng.randint(0, 2)),
        avoid_other=rng.choice(AVOID_OTHER) if rng.random() < 0.10 else "",
        loved=rng.sample(popular, rng.randint(0, 3)),
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
