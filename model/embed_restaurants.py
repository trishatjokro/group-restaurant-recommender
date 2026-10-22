"""
Embed every recommendable restaurant in a city into one "taste vector".

For each open restaurant with enough reviews, take its most recent LIKED reviews,
embed each with a sentence-transformer, and average them into one vector.
Liked reviews only, because low-star reviews are mostly about service (see EDA).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CITY = "philadelphia" # sample
FOLDER = Path("data") / "cities" / CITY

MODEL_NAME = "all-MiniLM-L6-v2"  # 384 numbers per text, reads up to 256 tokens
MIN_REVIEWS = 15                 # EDA decision: fewer reviews = unreliable vector
MAX_REVIEWS_USED = 50            # most recent liked reviews per restaurant
MIN_LIKED = 3                    # need a few liked reviews to describe a place
SHRINK = 3                       # pulls users with few reviews toward the overall average


def liked_reviews(reviews):
    """A review is 'liked' if it's above that user's own average rating.

    Most users wrote only 1 review, so their average IS that review and nothing
    would ever count as liked. Shrinking toward the overall average fixes that:
    a lone 5-star review still counts as liked, a lone 3-star one doesn't.
    """
    overall = reviews["stars"].mean()
    by_user = reviews.groupby("user_id")["stars"]
    user_avg = (by_user.transform("sum") + SHRINK * overall) / (by_user.transform("count") + SHRINK)
    return reviews[reviews["stars"] > user_avg]


def main():
    reviews = pd.read_parquet(FOLDER / "reviews_clean.parquet")
    restaurants = pd.read_parquet(FOLDER / "restaurants_clean.parquet")

    # Recommendable restaurants: open, with enough reviews.
    counts = reviews.groupby("business_id").size()
    restaurants["n_reviews"] = restaurants["business_id"].map(counts).fillna(0).astype(int)
    restaurants = restaurants[(restaurants["is_open"] == 1) & (restaurants["n_reviews"] >= MIN_REVIEWS)]

    # Most recent liked reviews of those restaurants.
    liked = liked_reviews(reviews)
    liked = liked[liked["business_id"].isin(restaurants["business_id"])]
    liked = liked.sort_values("date", ascending=False).groupby("business_id").head(MAX_REVIEWS_USED)
    n_liked = liked.groupby("business_id").size()
    keep = n_liked[n_liked >= MIN_LIKED].index
    liked = liked[liked["business_id"].isin(keep)]
    print(f"{len(keep):,} restaurants, embedding {len(liked):,} liked reviews")

    model = SentenceTransformer(MODEL_NAME)

    # Tokenization check: how many reviews are longer than the model can read?
    sample = liked["text"].sample(min(20_000, len(liked)), random_state=0)
    n_tokens = sample.apply(lambda t: len(model.tokenizer.tokenize(t)))
    print(f"tokens per review: median {n_tokens.median():.0f}, "
          f"{(n_tokens > model.max_seq_length).mean():.0%} cut off at {model.max_seq_length}")

    # Embed every review (normalized so cosine similarity = dot product).
    vectors = model.encode(liked["text"].tolist(), batch_size=256,
                           normalize_embeddings=True, show_progress_bar=True)

    # Average each restaurant's review vectors into one taste vector.
    liked = liked.reset_index(drop=True)
    rows = []
    for business_id, idx in liked.groupby("business_id").indices.items():
        v = vectors[idx].mean(axis=0)
        rows.append((business_id, len(idx), (v / np.linalg.norm(v)).tolist()))
    out = pd.DataFrame(rows, columns=["business_id", "n_reviews_used", "vector"])
    out = out.merge(restaurants[["business_id", "name", "categories", "stars",
                                 "n_reviews", "latitude", "longitude", "attributes"]],
                    on="business_id")

    out.to_parquet(FOLDER / "restaurant_vectors.parquet", index=False)
    print(f"Saved {len(out):,} restaurant vectors -> {FOLDER}/restaurant_vectors.parquet")


if __name__ == "__main__":
    main()
