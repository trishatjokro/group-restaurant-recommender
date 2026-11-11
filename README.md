# group-restaurant-recommender

DataSC Project: a group-based restaurant recommender. It turns Yelp reviews into
"taste vectors" with a sentence-transformer, combines group members' tastes,
and uses cosine similarity to find restaurants the whole group will like.

**Current city:** Philadelphia (7,021 recommendable restaurants, 1.15M cleaned reviews)

## Repo structure

```
├── make_sample.py          # builds the 50k-review sample used for cleaning + EDA
├── make_cities.py          # splits the full Yelp data into one dataset per city
├── pipeline.py             # runs everyone's cleaning functions (sample or a city)
├── make_users.py           # users + friend pairs for a city (for evaluation)
├── data/
│   ├── clean/              # cleaning functions (checks, text, duplicates, filters)
│   ├── output/             # cleaned sample
│   ├── *_sample.parquet    # raw sample
│   └── cities/<city>/      # full city data (NOT on GitHub; download from Drive)
├── EDA/                    # exploratory analysis scripts + charts
└── model/
    └── embed_restaurants.py  # one taste vector per restaurant
```

## Setup

1. Clone the repo and create a Python environment.
2. Install packages:
   `pip install pandas pyarrow langdetect matplotlib seaborn scipy scikit-learn folium sentence-transformers umap-learn plotly`
3. **Run everything from the repo's top folder.** In PyCharm: Edit Configurations →
   Edit configuration templates → Python → Working directory = the repo folder.

## Getting the data

The city data is too big for GitHub. Either:
- **Download from Drive** (easiest): put the files in `data/cities/philadelphia/`, or
- **Rebuild it yourself** from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
  (edit the `RAW =` path at the top of each script to where you saved it):

| Step | Command | Output |
|---|---|---|
| 1. Split by city | `python make_cities.py` | `data/cities/<city>/reviews.parquet`, `restaurants.parquet` |
| 2. Clean a city | set `CITY` in `pipeline.py`, then `python pipeline.py` | `reviews_clean.parquet`, `restaurants_clean.parquet` |
| 3. Users + friends | `python make_users.py` | `users.parquet`, `friend_pairs.parquet` |
| 4. Embed restaurants | `python model/embed_restaurants.py` | `restaurant_vectors.parquet` |

## Key decisions (from EDA)

- Only recommend **open** restaurants with **≥15 reviews**
- A review is **"liked"** if it's above that user's own average rating
- Restaurant vectors use **liked reviews only** (low-star reviews are mostly about service)
- Model: `all-MiniLM-L6-v2` (384-dim); long reviews are cut at 256 tokens (only 6% are)
- Results show **one location per chain**
