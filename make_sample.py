# Sample data for cleaning and EDA

import json
from pathlib import Path

import pandas as pd

# Where YOU downloaded the Yelp Open Dataset (outside the repo). Change this.
RAW = Path.home() / "Documents" / "Data SC" / "Yelp JSON" / "yelp_dataset"
OUT = Path("data")

TARGET_REVIEWS = 50_000
SEED = 42


def main():
    biz = pd.read_json(RAW / "yelp_academic_dataset_business.json", lines=True)
    restaurants = biz[biz["categories"].str.contains("Restaurants", na=False)].copy()
    # Store attributes/hours as JSON text so parquet/CSV can hold them as-is.
    for col in ("attributes", "hours"):
        restaurants[col] = restaurants[col].apply(lambda v: json.dumps(v) if v is not None else None)

    # Random cluster sample: shuffle restaurants, take them until ~50k reviews.
    shuffled = restaurants.sample(frac=1, random_state=SEED)
    sampled = shuffled[shuffled["review_count"].cumsum() <= TARGET_REVIEWS]
    ids = set(sampled["business_id"])

    rows = []
    with open(RAW / "yelp_academic_dataset_review.json", "rb") as f:
        for line in f:
            d = json.loads(line)
            if d["business_id"] in ids:
                rows.append(d)
    reviews = pd.DataFrame(rows)

    OUT.mkdir(exist_ok=True)
    restaurants.to_parquet(OUT / "restaurants_all.parquet", index=False)
    sampled.to_parquet(OUT / "restaurants_sample.parquet", index=False)
    reviews.to_parquet(OUT / "reviews_sample.parquet", index=False)
    print(f"Sample: {len(sampled):,} restaurants, {len(reviews):,} reviews -> {OUT}/")


if __name__ == "__main__":
    main()