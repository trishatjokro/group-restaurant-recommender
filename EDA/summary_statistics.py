import pandas as pd
import os

print(os.getcwd())

df_restaurants=pd.read_parquet("data/output/restaurants_clean.parquet")
df_reviews=pd.read_parquet("data/output/reviews_clean.parquet")

#descriptive stats
restaurants_stat =df_restaurants[["stars", "review_count"]].describe()
review_stat=df_reviews[["stars", "useful","funny","cool"]].describe()

print("Descriptive statistics: ")
print(restaurants_stat)
print(review_stat)

#missing stats
def missing(df):
    return pd.DataFrame({
    "missing_count": df.isna().sum(),
    "missing_percentage": df.isna().mean() * 100
})
print(missing(df_restaurants))
print(missing(df_reviews))

print(" ")
print("Restaurants per city: ")
def restaurants_per_city(df):
    restaurants = df.groupby("city")["name"].agg(
        restaurants_count="count", #count resto per city
        restaurants=", ".join)
    return restaurants.reset_index()
print(restaurants_per_city(df_restaurants))








