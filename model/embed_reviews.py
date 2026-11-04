from pathlib import Path 
import pandas as pd 
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "output"

MODEL_NAME = "all-MiniLM-L6-v2"

def main(): 
    # load the cleaned reviews
    reviews = pd.read_parquet(FOLDER / "reviews_clean.parquet")
    print(f"Embedding {len(reviews):,} reviews")

    # load the embedding model 
    model = SentenceTransformer(MODEL_NAME)

    # turn every review into a vector 
    vectors = model.encode(
        reviews["text"].fillna("").tolist(), 
        batch_size = 256, 
        normalize_embeddings = True, 
        show_progress_bar = True
    )

    print("Vector shape: ", vectors.shape)

    # keep useful identifying information 
    out = reviews[
        ["review_id", "user_id", "business_id", "stars", "date"]
    ].copy() 

    # add the vector to each review 
    out["vector"] = vectors.tolist() 

    # save everything 
    output_file = FOLDER / "review_vectors.parquet"
    out.to_parquet(output_file, index = False) 
    print(f"Saved {len(out):,} review vectors -> {output_file}")

if __name__ == "__main__": 
    main() 