"""
Results:
3,064 restaurants are outside every city and are skipped
philadelphia    14,476 restaurants  1,195,479 reviews
tampa            8,136 restaurants    765,544 reviews
indianapolis     4,150 restaurants    335,843 reviews
nashville        4,353 restaurants    434,726 reviews
tucson           2,672 restaurants    267,048 reviews
new_orleans      3,640 restaurants    558,340 reviews
edmonton         2,410 restaurants     68,882 reviews
saint_louis      5,232 restaurants    391,499 reviews
reno             1,673 restaurants    243,613 reviews
boise            1,302 restaurants    100,065 reviews
santa_barbara    1,160 restaurants    211,722 reviews
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# This is for my personal path. Change it to fit your own.
RAW = Path.home() / "Documents" / "Data SC" / "Yelp JSON" / "yelp_dataset"
OUT = Path("data") / "cities"

RADIUS_KM = 40  # how far from the centre counts as "the city"

# City centres (lat, lon). Distance is used instead of the city column because
# city names are messy ("Phila", suburbs, typos).
CENTERS = {
    "philadelphia": (39.9526, -75.1652),
    "tampa": (27.9506, -82.4572),
    "indianapolis": (39.7684, -86.1581),
    "nashville": (36.1627, -86.7816),
    "tucson": (32.2226, -110.9747),
    "new_orleans": (29.9511, -90.0715),
    "edmonton": (53.5461, -113.4938),
    "saint_louis": (38.6270, -90.1994),
    "reno": (39.5296, -119.8138),
    "boise": (43.6150, -116.2023),
    "santa_barbara": (34.4208, -119.6982),
}


def distance_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main():
    biz = pd.read_json(RAW / "yelp_academic_dataset_business.json", lines=True)
    restaurants = biz[biz["categories"].str.contains("Restaurants", na=False)].copy()
    for col in ("attributes", "hours"):
        restaurants[col] = restaurants[col].apply(lambda v: json.dumps(v) if v is not None else None)

    # Label each restaurant with the city it falls in (if any).
    restaurants["metro"] = None
    for city, center in CENTERS.items():
        dist = distance_km(restaurants["latitude"], restaurants["longitude"], *center)
        restaurants.loc[dist <= RADIUS_KM, "metro"] = city
    print(f"{restaurants['metro'].isna().sum():,} restaurants are outside every city and are skipped")
    restaurants = restaurants.dropna(subset=["metro"])
    city_of = dict(zip(restaurants["business_id"], restaurants["metro"]))

    # Big review file, sorting reviews into cities
    rows = {city: [] for city in CENTERS}
    with open(RAW / "yelp_academic_dataset_review.json", "rb") as f:
        for line in f:
            d = json.loads(line)
            city = city_of.get(d["business_id"])
            if city:
                rows[city].append(d)

    for city in CENTERS:
        folder = OUT / city
        folder.mkdir(parents=True, exist_ok=True)
        city_restaurants = restaurants[restaurants["metro"] == city].drop(columns="metro")
        city_reviews = pd.DataFrame(rows.pop(city))
        city_restaurants.to_parquet(folder / "restaurants.parquet", index=False)
        city_reviews.to_parquet(folder / "reviews.parquet", index=False)
        print(f"{city:15s} {len(city_restaurants):>6,} restaurants {len(city_reviews):>10,} reviews")


if __name__ == "__main__":
    main()
