"""
Palate API: the app sends a group's answers, we return the top restaurants.
Two ranking methods (pick with "method" in the request):
  "hybrid"  (default) hybrid_model.py: each person's score = 50/50 blend of their SVM
            and cosine scores (percentile ranks) -> consensus (average minus
            disagreement, from group_recommender.py)
  "average" taste_profile.py (each person) + group_vector.py (average group vector)
Run from the repo's top folder:  python3 -m uvicorn api.app:app --port 8000

Dietary needs are a MUST (only restaurants tagged with that diet), "avoid" is a soft
preference: in "hybrid" it trains the SVM's negatives; in "average" unusually close
places are pushed down (never removed). If too few places
pass the filters, diets and then budgets are relaxed, the same way the app's
on-device fallback does it.
"""

import unicodedata
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model.group_vector import group_vector
from model.hybrid_model import hybrid_scores
from model.taste_profile import allowed, build_taste_profile, score_restaurants

VECTORS = Path("data") / "output" / "restaurant_vectors.parquet"        # all 7,021 Philly restaurants
ADDRESSES = Path("philly_map") / "restaurants_clean.parquet"           # has an address for every one
AVOID_WEIGHT = 1.0       # how hard "avoid" pushes places down (tested Nov 16: removes avoided places, keeps the rest)
NO_DIET = "none"
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
restaurants = pd.read_parquet(VECTORS)
addresses = pd.read_parquet(ADDRESSES, columns=["business_id", "address"])
restaurants = restaurants.merge(addresses, on="business_id", how="left")
V = np.vstack(restaurants["vector"].values)
price = pd.to_numeric(restaurants["attributes"].str.extract(r'"RestaurantsPriceRange2": "(\d)"')[0], errors="coerce")

app = FastAPI(title="Palate API")
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://localhost:3000",     # React dev servers
                                  "http://localhost:8081", "http://127.0.0.1:8081",     # Expo web (palate/)
                                  "http://localhost:19006"],
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
    diets: list[str] = []          # e.g. ["halal"]; always treated as "must"


class Request(BaseModel):
    members: list[Member]
    top_k: int = 5
    method: Literal["hybrid", "average"] = "hybrid"


def percentile(scores, ok):
    """0–100 match %: 100 = as good as the best allowed restaurant, 50 = a typical one.
    (A plain rank percentile put every top result at ~100%, which told users nothing.)"""
    best, typical = scores[ok].max(), np.median(scores[ok])
    pct = 50 + 50 * (scores - typical) / max(best - typical, 1e-9)
    return np.clip(pct, 0, 100)


def chain_key(name):
    """Same chain at different locations ("Nudy's Café - West Chester" / "Nudy's Cafe - Chadds Ford")."""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return n.split(" - ")[0].strip()


def allowed_for_group(profiles, relax):
    """'strict' = budgets + must-diets; 'relaxDiet' = budgets only; 'relaxAll' = everything."""
    if relax == "relaxAll":
        return np.ones(len(restaurants), dtype=bool)
    ok = np.ones(len(restaurants), dtype=bool)
    for p in profiles:
        q = dict(p, diets_must=set()) if relax == "relaxDiet" else p
        ok &= allowed(q, restaurants)
    return ok


def avoid_text(m):
    return ", ".join([a.replace(" ($$$)", "") for a in m.avoid] + ([m.avoid_other] if m.avoid_other else []))


def diets_of(m):
    return {d.lower(): "must" for d in m.diets if d.lower() != NO_DIET}


def build_profiles(members):
    """One taste profile per member (taste_profile.py). Used for filters in both methods."""
    profiles = []
    for m in members:
        if m.skipped:   # no taste picks, but their budget / diet / avoids still count
            profiles.append(build_taste_profile(restaurants, dislikes_text=avoid_text(m),
                                                prices=m.prices, diets=diets_of(m)))
            continue
        profiles.append(build_taste_profile(restaurants, loves_text=m.loves_text, bubbles=m.bubbles,
                                            loved_names=m.loved_restaurants, dislikes_text=avoid_text(m),
                                            prices=m.prices, diets=diets_of(m)))
    return profiles


def eligible_rows(profiles, top_k):
    """Everyone's filters, relaxed step by step until at least top_k places fit."""
    visited = restaurants["business_id"].isin(set().union(*[p["visited"] for p in profiles])).values
    for relaxed in ("strict", "relaxDiet", "relaxAll"):
        ok = allowed_for_group(profiles, relaxed) & ~visited
        if ok.sum() >= top_k:
            break
    return ok, relaxed


def score_average(members, profiles, ok):
    """Average group vector (group_vector.py) + "avoid" push-down."""
    combined = group_vector(profiles, restaurants)
    g, group_dislikes = combined["vector"], combined["dislikes"]
    score = V @ g
    if group_dislikes is not None:              # "avoid": push down only places unusually close to it
        dsim = V @ group_dislikes
        score = score - AVOID_WEIGHT * np.maximum(0, dsim - np.median(dsim[ok]))
    member_scores = {m.name: score_restaurants(p, restaurants)
                     for m, p in zip(members, profiles) if p["likes"] is not None}
    return score, member_scores


def score_hybrid(members):
    """hybrid_model.py: per-person SVM + cosine blend, combined by consensus.
    Members are passed in the mock-user format the model was built and tested with
    (user_id = the member's name; the app makes names unique)."""
    users = []
    for m in members:
        if m.skipped:
            users.append({"user_id": m.name, "prices": m.prices, "diets": diets_of(m),
                          "dislikes_text": avoid_text(m)})
            continue
        users.append({"user_id": m.name, "cravings": m.bubbles, "specific": [],
                      "loves_text": m.loves_text, "loved_restaurants": m.loved_restaurants,
                      "dislikes_text": avoid_text(m), "prices": m.prices, "diets": diets_of(m)})
    res = hybrid_scores(users, restaurants)     # one row per restaurant, same order as `restaurants`
    member_scores = {m.name: spread(res[f"{m.name}_hybrid"].to_numpy())
                     for m in members if f"{m.name}_hybrid" in res.columns}
    return spread(res["consensus"].to_numpy()), member_scores


def spread(rank_scores):
    """hybrid_model's scores are blends of rank percentiles (0-1), so the top ~1% of 7,021 places
    all sit at 0.99+ and would all show "100%". Map them onto a bell curve (z-scores)
    so the top results spread out. Same order, only the match % changes."""
    n = len(rank_scores)
    return norm.ppf(np.clip(rank_scores, 0.5 / n, 1 - 0.5 / n))


@app.post("/recommend")
def recommend(req: Request):
    members = [m for m in req.members]
    profiles = build_profiles(members)
    ok, relaxed = eligible_rows(profiles, req.top_k)

    if req.method == "hybrid":
        score, member_scores = score_hybrid(members)
    else:
        score, member_scores = score_average(members, profiles, ok)
    group_pct = percentile(score, ok)
    member_pct = {name: percentile(sc, ok) for name, sc in member_scores.items()}

    results, seen = [], set()
    for j in np.argsort(-np.where(ok, score, -np.inf)):
        if not ok[j]:
            break
        key = chain_key(restaurants.at[j, "name"])
        if key in seen:
            continue
        seen.add(key)
        cats = restaurants.at[j, "categories"] or ""
        fits = {m.name: [b for b in m.bubbles if b in BUBBLE_TAGS and pd.Series([cats]).str.contains(BUBBLE_TAGS[b]).iloc[0]]
                for m in members if not m.skipped}
        matched = {n: bs for n, bs in fits.items() if bs}
        tapped = sorted({b for m in members for b in m.bubbles})
        fit = {n: int(round(member_pct[n][j])) for n in member_pct}
        results.append({
            "id": restaurants.at[j, "business_id"],
            "name": restaurants.at[j, "name"], "categories": cats,
            "categories_list": [c.strip() for c in cats.split(",") if c.strip() and c.strip() != "Restaurants"],
            "price": "$" * int(price[j]) if not np.isnan(price[j]) else None,
            "price_level": None if np.isnan(price[j]) else int(price[j]),
            "stars": float(restaurants.at[j, "stars"]), "address": restaurants.at[j, "address"],
            "reviews": int(restaurants.at[j, "n_reviews"]),
            "lat": float(restaurants.at[j, "latitude"]), "lng": float(restaurants.at[j, "longitude"]),
            "match_pct": int(round(group_pct[j])),
            "best_fit_for": max(fit, key=fit.get) if fit else None,
            "member_fit": fit,
            "reason": "; ".join(f"Fits {n}'s picks: {', '.join(bs)}" for n, bs in matched.items()) or None,
            "label": (f"Not a {' / '.join(tapped)} restaurant, but reviewers describe similar dishes"
                      if tapped and not matched else None),
        })
        if len(results) == req.top_k:
            break
    return {"results": results, "relaxed": relaxed, "method": req.method}


@app.get("/search")
def search(q: str, limit: int = 8):
    hits = restaurants.loc[restaurants["name"].str.contains(q, case=False, regex=False), "name"]
    return {"names": hits.drop_duplicates().head(limit).tolist()}


@app.get("/health")
def health():
    return {"ok": True}
