# Turn a person's initial answers into a taste profile, and score every
# restaurant for that person.


import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # same model used for the restaurant vectors
DISLIKE_WEIGHT = 0.5              # how strongly dislikes push a restaurant down

DIET_CATEGORIES = {               # Yelp category names to look for
    "vegetarian": "Vegetarian",
    "vegan": "Vegan",
    "halal": "Halal",
    "gluten-free": "Gluten-Free",
}

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _normalize(v):
    return v / np.linalg.norm(v)


def build_taste_profile(restaurants, loves_text="", loved_names=(), dislikes_text="",
                        prices=(), diets=()):
    """restaurants: the restaurant_vectors table. Returns a dict describing one person."""
    model = get_model()
    V = np.vstack(restaurants["vector"].values)

    like_parts = []
    if loves_text.strip():
        like_parts.append(model.encode(loves_text, normalize_embeddings=True))
    picked = restaurants.index[restaurants["name"].isin(loved_names)]
    like_parts += [V[i] for i in picked]

    return {
        "likes": _normalize(np.mean(like_parts, axis=0)) if like_parts else None,
        "dislikes": (model.encode(dislikes_text, normalize_embeddings=True)
                     if dislikes_text.strip() else None),
        "visited": set(restaurants.loc[picked, "business_id"]),
        "prices": set(prices),     # e.g. {1, 2} for $ and $$; empty = any price
        "diets": set(diets),       # e.g. {"vegetarian"}; empty = no restriction
    }


def score_restaurants(taste_profile, restaurants):
    """One score per restaurant for this person (higher = better match)."""
    V = np.vstack(restaurants["vector"].values)
    scores = np.zeros(len(restaurants))
    if taste_profile["likes"] is not None:
        scores += V @ taste_profile["likes"]
    if taste_profile["dislikes"] is not None:
        scores -= DISLIKE_WEIGHT * (V @ taste_profile["dislikes"])
    return scores


def allowed(taste_profile, restaurants):
    """True for restaurants that fit this person's price range and dietary needs."""
    ok = np.ones(len(restaurants), dtype=bool)
    if taste_profile["prices"]:
        price = restaurants["attributes"].str.extract(r'"RestaurantsPriceRange2": "(\d)"')[0]
        price = pd.to_numeric(price, errors="coerce")
        ok &= price.isin(taste_profile["prices"]).values | price.isna().values  # unknown price = keep
    for diet in taste_profile["diets"]:
        ok &= restaurants["categories"].str.contains(DIET_CATEGORIES[diet], na=False).values
    return ok
