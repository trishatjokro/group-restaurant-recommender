import pandas as pd

def clean_duplicates(df, restaurants_all):
    df = df.copy()

    #remove duplicate reviews
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    after = len(df)
    num_duplicates = before - after 

    #keep newest review for users reviewing same restaurant twice 
    before = after
    df = (
        df
        .sort_values("date", ascending=False)
        .drop_duplicates(subset=["user_id", "business_id"], keep="first")
    )
    after = len(df)
    num_duplicate_users = before - after

    #remove restaurants not in dataset 
    invalid = ~df["business_id"].isin(restaurants_all["business_id"])
    num_invalid = invalid.sum()
    df = df[~invalid].reset_index(drop=True)
    
    print(
        f"clean_duplicates: removed {num_duplicates} duplicate reviews, "
        f"{num_duplicate_users} duplicate users reviewing same restaurant, "
        f"{num_invalid} reviews with invalid restaurant ids"
    )
    return df

if __name__ == "__main__":
    restaurants_all = pd.read_parquet("../restaurants_all.parquet")
    restaurants_sample = pd.read_parquet("../restaurants_sample.parquet")
    reviews = pd.read_parquet("../reviews_sample.parquet")
    print(restaurants_all.shape)
    print(reviews.shape)
    print(reviews.columns)
    reviews.info()
    print(reviews.duplicated(subset=["text"]).sum())
    clean_duplicates(reviews, restaurants_all)
    

