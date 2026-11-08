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
DIET_PHRASES = {                  # added to likes when a diet is "prefer"
    "vegetarian": "great vegetarian options",
    "vegan": "great vegan options",
    "halal": "halal food",
    "gluten-free": "gluten-free options",
}
DIET_BONUS = 0.03                 # small boost for tagged places when a diet is "prefer"

# Each tapped bubble is embedded as its own short description (experiment 11):
# richer text matches better than one word, and embedding each pick separately
# stops someone who taps many bubbles from being drowned out.
BUBBLE_DESCRIPTIONS = {
    # cravings (cuisines)
    "French": "French food: crepes, croissants, steak frites and bistro dishes",
    "Japanese": "Japanese food: sushi, ramen, udon and tempura",
    "Thai": "Thai food: pad thai, green curry and tom yum",
    "Korean": "Korean food: Korean BBQ, bibimbap and kimchi",
    "Indian": "Indian food: curry, naan, tikka masala and biryani",
    "American (New)": "New American restaurant with seasonal small plates",
    "Italian": "Italian food: pasta, risotto and pizza",
    "American (Traditional)": "classic American food: burgers, fries and comfort food",
    "Mediterranean": "Mediterranean food: hummus, falafel, gyros and kebabs",
    "Chinese": "Chinese food: dumplings, lo mein, fried rice and dim sum",
    "Vietnamese": "Vietnamese food: pho, banh mi and vermicelli",
    "Mexican": "Mexican food: tacos, burritos and guacamole",
    # anything specific (food types)
    "Sandwiches": "sandwiches, hoagies and subs",
    "Pizza": "pizza by the slice or whole pie",
    "Breakfast & Brunch": "breakfast and brunch: pancakes, eggs benedict and omelets",
    "Fast Food": "quick fast food and drive-thru",
    "Burgers": "juicy burgers and fries",
    "Coffee & Tea": "coffee shop with lattes, espresso and tea",
    "Seafood": "seafood: oysters, crab, shrimp and fresh fish",
    "Cafes": "cozy cafe with coffee and pastries",
    "Salad": "fresh salads and grain bowls",
    "Delis": "deli with pastrami, corned beef and hoagies",
    "Chicken": "fried chicken and chicken wings",
    "Cheesesteak": "Philly cheesesteaks",
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
                        prices=(), diets=None, bubbles=()):
    """restaurants: the restaurant_vectors table. Returns a dict describing one person.
    loves_text: anything the person TYPED (e.g. "I'm craving spicy noodles").
    bubbles: the bubbles they TAPPED (e.g. ["Japanese", "Pizza"]); each is embedded
             separately using BUBBLE_DESCRIPTIONS. Don't also put tapped bubbles in loves_text.
    diets: e.g. {"halal": "must", "vegan": "prefer"}; empty = no restriction."""
    model = get_model()
    V = np.vstack(restaurants["vector"].values)

    like_parts = []
    if loves_text.strip():
        like_parts.append(model.encode(loves_text, normalize_embeddings=True))
    for bubble in bubbles:
        like_parts.append(model.encode(BUBBLE_DESCRIPTIONS.get(bubble, bubble),
                                       normalize_embeddings=True))
    picked = restaurants.index[restaurants["name"].isin(loved_names)]
    like_parts += [V[i] for i in picked]
    diets = diets or {}
    for diet, mode in diets.items():
        if mode == "prefer":
            like_parts.append(model.encode(DIET_PHRASES[diet], normalize_embeddings=True))

    return {
        "likes": _normalize(np.mean(like_parts, axis=0)) if like_parts else None,
        "dislikes": (model.encode(dislikes_text, normalize_embeddings=True)
                     if dislikes_text.strip() else None),
        "visited": set(restaurants.loc[picked, "business_id"]),
        "prices": set(prices),     # e.g. {1, 2} for $ and $$; empty = any price
        "diets_must": {d for d, m in diets.items() if m == "must"},
        "diets_prefer": {d for d, m in diets.items() if m == "prefer"},
    }


def score_restaurants(taste_profile, restaurants):
    """One score per restaurant for this person (higher = better match)."""
    V = np.vstack(restaurants["vector"].values)
    scores = np.zeros(len(restaurants))
    if taste_profile["likes"] is not None:
        scores += V @ taste_profile["likes"]
    if taste_profile["dislikes"] is not None:
        scores -= DISLIKE_WEIGHT * (V @ taste_profile["dislikes"])
    for diet in taste_profile["diets_prefer"]:
        tagged = restaurants["categories"].str.contains(DIET_CATEGORIES[diet], na=False).values
        scores += DIET_BONUS * tagged
    return scores


def allowed(taste_profile, restaurants):
    """True for restaurants that fit this person's price range and dietary needs."""
    ok = np.ones(len(restaurants), dtype=bool)
    if taste_profile["prices"]:
        price = restaurants["attributes"].str.extract(r'"RestaurantsPriceRange2": "(\d)"')[0]
        price = pd.to_numeric(price, errors="coerce")
        ok &= price.isin(taste_profile["prices"]).values | price.isna().values  # unknown price = keep
    for diet in taste_profile["diets_must"]:
        ok &= restaurants["categories"].str.contains(DIET_CATEGORIES[diet], na=False).values
    return ok
