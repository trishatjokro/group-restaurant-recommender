from pathlib import Path

import pandas as pd

from data.clean.checks import check_stars, check_date, check_cities, clean_city
from data.clean.text import text_cleaning
from data.clean.duplicates import clean_duplicates
from data.clean.filters import filter_reviews

DATA = Path("data")
OUTPUT = DATA / "output"


def main():
    reviews = pd.read_parquet(DATA / "reviews_sample.parquet")
    restaurants = pd.read_parquet(DATA / "restaurants_sample.parquet")
    restaurants_all = pd.read_parquet(DATA / "restaurants_all.parquet")
    print(f"Loaded {len(reviews):,} reviews, {len(restaurants):,} restaurants\n")

    # --- Reviews -----------------------------------------------------------
    n_start = len(reviews)

    check_stars(reviews)                          # 1. Gisella: report bad stars/dates
    check_date(reviews)
    reviews = text_cleaning(reviews)              # 2. Anshika: fix the text
    reviews = clean_duplicates(reviews, restaurants_all)  # 3. Connor: after text, so near-copies match
    reviews = filter_reviews(reviews)             # 4. Evelyn: length/language on cleaned text

    # --- Restaurants -------------------------------------------------------
    check_cities(restaurants)
    restaurants = clean_city(restaurants)         # Gisella: adds city_clean

    # --- Save --------------------------------------------------------------
    OUTPUT.mkdir(exist_ok=True)
    reviews.to_parquet(OUTPUT / "reviews_clean.parquet", index=False)
    restaurants.to_parquet(OUTPUT / "restaurants_clean.parquet", index=False)

    print(f"\nReviews: {n_start:,} -> {len(reviews):,} "
          f"({n_start - len(reviews):,} removed)")
    print(f"Saved to {OUTPUT}/")


if __name__ == "__main__":
    main()