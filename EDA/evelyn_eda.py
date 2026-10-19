import pandas as pd
import json
import matplotlib.pyplot as plt

restaurants = pd.read_parquet("data/output/restaurants_clean.parquet")
reviews = pd.read_parquet("data/output/reviews_clean.parquet")

# rating vs price 
def get_price(attributes):
    if pd.isna(attributes):
        return None

    try:
        attributes = json.loads(attributes)
        return attributes.get("RestaurantsPriceRange2")
    except:
        return None

restaurants["price"] = restaurants["attributes"].apply(get_price)
restaurants["price"] = pd.to_numeric(restaurants["price"], errors="coerce")

print(restaurants[["name", "stars", "price"]].head(20))
print(restaurants["price"].value_counts().sort_index())

rating_price = (
    restaurants
    .dropna(subset=["price", "stars"])
    .groupby("price")["stars"]
    .agg(["mean", "median", "count"])
)

print("\nRating vs. Price:")
print(rating_price)

plt.figure()
rating_price["mean"].plot(kind="bar")

rating_price_data = restaurants[["price", "stars"]].dropna()

correlation = rating_price_data["price"].corr(
    rating_price_data["stars"],
    method="spearman"
)

print("Spearman correlation:", correlation)

plt.title("Average Restaurant Rating by Price Level")
plt.xlabel("Price Level")
plt.ylabel("Average Rating")
plt.xticks(
    ticks=[0, 1, 2, 3],
    labels=[r"\$", r"\$\$", r"\$\$\$", r"\$\$\$\$"],
    rotation=0
)
plt.tight_layout()

# review length vs rating 
reviews["review_length"] = reviews["text"].str.len()

length_rating = (
    reviews
    .groupby("stars")["review_length"]
    .agg(["mean", "median", "count"])
)

print("\nReview Length vs. Rating:")
print(length_rating)

length_rating_data = reviews[["review_length", "stars"]].dropna()

correlation = length_rating_data["review_length"].corr(
    length_rating_data["stars"],
    method="spearman"
)

print("Spearman correlation:", correlation)

plt.figure()
length_rating["mean"].plot(kind="bar")

plt.title("Average Review Length by Rating")
plt.xlabel("Rating")
plt.ylabel("Average Review Length (characters)")
plt.xticks(rotation=0)
plt.tight_layout()

# cuisine vs price 
cuisines = [
    "American (New)",
    "American (Traditional)",
    "Chinese",
    "Italian",
    "Mexican",
    "Japanese",
    "Thai",
    "Indian",
    "Mediterranean",
    "French",
    "Vietnamese",
    "Korean"
]

def get_cuisine(categories):
    if pd.isna(categories):
        return None

    category_list = [category.strip() for category in categories.split(",")]

    for cuisine in cuisines:
        if cuisine in category_list:
            return cuisine

    return None

restaurants["cuisine"] = restaurants["categories"].apply(get_cuisine)

cuisine_price = (
    restaurants
    .dropna(subset=["cuisine", "price"])
    .groupby("cuisine")["price"]
    .agg(["mean", "median", "count"])
    .sort_values("mean")
)

print("\nCuisine vs. Price:")
print(cuisine_price)

plt.figure()
cuisine_price["mean"].plot(kind="barh")

plt.title("Average Price Level by Cuisine")
plt.xlabel("Average Price Level")
plt.ylabel("Cuisine")
plt.tight_layout()
plt.show()