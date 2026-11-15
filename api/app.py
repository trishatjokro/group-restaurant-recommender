"""
Palate API: the React app sends a group's answers, we return the top restaurants.
Uses taste_profile.py (each person) + group_vector.py (average group vector).
Run from the repo's top folder:  .venv/bin/python -m uvicorn api.app:app --port 8000
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.group_vector import group_vector
from model.taste_profile import allowed, build_taste_profile, score_restaurants

DATA = Path("data") / "cities" / "philadelphia"
BUBBLE_TAGS = {  # Yelp categories that count as "fits" a bubble (for reasons/labels only)
    "French": "French", "Japanese": "Japanese|Sushi|Ramen", "Thai": "Thai", "Korean": "Korean",
    "Indian": "Indian|Pakistani|Himalayan", "American (New)": r"American \(New\)", "Italian": "Italian",
    "American (Traditional)": r"American \(Traditional\)", "Mediterranean": "Mediterranean|Greek|Middle Eastern",
    "Chinese": "Chinese|Szechuan|Cantonese|Dim Sum", "Vietnamese": "Vietnamese", "Mexican": "Mexican|Tacos",
    "Sandwiches": "Sandwiches", "Pizza": "Pizza", "Breakfast & Brunch": "Breakfast & Brunch",
    "Fast Food": "Fast Food", "Burgers": "Burgers", "Coffee & Tea": "Coffee & Tea|Cafes", "Seafood": "Seafood",
    "Cafes": "Cafes|Coffee & Tea", "Salad": "Salad", "Delis": "Delis", "Chicken": "Chicken",
    "Cheesesteak": "Cheesesteaks",
}

# Load once at startup (~10 s), not on every request.
restaurants = pd.read_parquet(DATA / "restaurant_vectors.parquet")
addresses = pd.read_parquet(DATA / "restaurants_clean.parquet", columns=["business_id", "address"])
restaurants = restaurants.merge(addresses, on="business_id", how="left")
V = np.vstack(restaurants["vector"].values)
price = pd.to_numeric(restaurants["attributes"].str.extract(r'"RestaurantsPriceRange2": "(\d)"')[0], errors="coerce")

app = FastAPI(title="Palate API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])


class Member(BaseModel):
    name: str
    skipped: bool = False
    bubbles: list[str] = []
    loves_text: str = ""
    avoid: list[str] = []
    avoid_other: str = ""
    loved_restaurants: list[str] = []
    prices: list[int] = []


class Request(BaseModel):
    members: list[Member]
    top_k: int = 5


def percentile(scores, ok):
    """0–100: how high each restaurant ranks among the allowed ones."""
    s = np.where(ok, scores, -np.inf)
    return (s.argsort().argsort() - (~ok).sum()) / max(ok.sum() - 1, 1) * 100


@app.post("/recommend")
def recommend(req: Request):
    members = [m for m in req.members]
    profiles = []
    for m in members:
        if m.skipped:
            profiles.append(build_taste_profile(restaurants))
            continue
        avoid = ", ".join([a.replace(" ($$$)", "") for a in m.avoid] + ([m.avoid_other] if m.avoid_other else []))
        profiles.append(build_taste_profile(restaurants, loves_text=m.loves_text, bubbles=m.bubbles,
                                            loved_names=m.loved_restaurants, dislikes_text=avoid,
                                            prices=m.prices))
    g = group_vector(profiles, restaurants)["vector"]

    ok = np.logical_and.reduce([allowed(p, restaurants) for p in profiles])
    ok &= ~restaurants["business_id"].isin(set().union(*[p["visited"] for p in profiles])).values
    group_pct = percentile(V @ g, ok)
    member_pct = {m.name: percentile(score_restaurants(p, restaurants), ok)
                  for m, p in zip(members, profiles) if p["likes"] is not None}

    results, seen = [], set()
    for j in np.argsort(-np.where(ok, V @ g, -np.inf)):
        if not ok[j] or restaurants.at[j, "name"] in seen:
            continue
        seen.add(restaurants.at[j, "name"])
        cats = restaurants.at[j, "categories"] or ""
        fits = {m.name: [b for b in m.bubbles if b in BUBBLE_TAGS and pd.Series([cats]).str.contains(BUBBLE_TAGS[b]).iloc[0]]
                for m in members if not m.skipped}
        matched = {n: bs for n, bs in fits.items() if bs}
        tapped = sorted({b for m in members for b in m.bubbles})
        fit = {n: int(round(member_pct[n][j])) for n in member_pct}
        results.append({
            "name": restaurants.at[j, "name"], "categories": cats,
            "price": "$" * int(price[j]) if not np.isnan(price[j]) else None,
            "stars": float(restaurants.at[j, "stars"]), "address": restaurants.at[j, "address"],
            "match_pct": int(round(group_pct[j])),
            "best_fit_for": max(fit, key=fit.get) if fit else None,
            "member_fit": fit,
            "reason": "; ".join(f"Fits {n}'s picks: {', '.join(bs)}" for n, bs in matched.items()) or None,
            "label": (f"Not a {' / '.join(tapped)} restaurant, but reviewers describe similar dishes"
                      if tapped and not matched else None),
        })
        if len(results) == req.top_k:
            break
    return {"results": results}


@app.get("/search")
def search(q: str, limit: int = 8):
    hits = restaurants.loc[restaurants["name"].str.contains(q, case=False, regex=False), "name"]
    return {"names": hits.drop_duplicates().head(limit).tolist()}


@app.get("/health")
def health():
    return {"ok": True}
