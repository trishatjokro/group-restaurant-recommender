# SETTING UP

import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sns.set_theme(style="whitegrid")

REVIEWS_PATH  = "data/reviews.parquet"
STARS_COL     = "stars"
TEXT_COL      = "text"
BIZ_ID_COL    = "business_id"
CUISINE_COL   = "cuisine"
SAMPLE_SIZE   = 200_000                   

reviews = pd.read_parquet("data/output/reviews_clean.parquet")

reviews["star_group"] = reviews[STARS_COL].round()
reviews = reviews.dropna(subset=["star_group"]).assign(star_group=lambda d: d["star_group"].astype(int))
print(reviews.shape)

# STAR RATINGS

star_counts = reviews["star_group"].value_counts().sort_index()
ax = star_counts.plot(kind="bar", color="steelblue", figsize=(6, 4))
ax.set(title="Star rating distribution", xlabel="Stars", ylabel="Number of reviews")
plt.xticks(rotation=0)
plt.show()

# REVIEW LENGTH

reviews["word_count"] = reviews[TEXT_COL].fillna("").str.split().str.len()
cap = reviews["word_count"].quantile(0.99)      
median = reviews["word_count"].median()

plt.figure(figsize=(8, 4))
sns.histplot(reviews.loc[reviews["word_count"] <= cap, "word_count"], bins=50, color="steelblue")
plt.axvline(median, color="red", linestyle="--", label=f"median = {median:.0f} words")
plt.title("Review length (words, top 1% trimmed)")
plt.xlabel("Words per review")
plt.legend()
plt.show()

# MOST COMMON WORDS BY STAR RATING

stars = sorted(sample["star_group"].unique())
fig, axes = plt.subplots(1, len(stars), figsize=(4 * len(stars), 5))
for ax, s in zip(np.atleast_1d(axes), stars):
    words, counts = zip(*top_words(sample.loc[sample["star_group"] == s, TEXT_COL]))
    sns.barplot(x=list(counts), y=list(words), ax=ax, color="steelblue")
    ax.set_title(f"{s}-star reviews")
    ax.set_xlabel("Count")
plt.tight_layout()
plt.show()

# TOP CUISINES

RESTAURANTS_PATH = "data/output/restaurants_clean.parquet"
REST_ID_COL      = "business_id"   
CUISINE_COL      = "categories"   

rest = pd.read_parquet("data/output/restaurants_clean.parquet")

# generic labels that aren't cuisines - can edit depending on what counts as "cuisine"
GENERIC = {"restaurants", "food", "nightlife", "bars", "event planning & services",
           "shopping", "local services", "arts & entertainment", "specialty food"}

def to_list(x):
    if isinstance(x, str):
        return [c.strip() for c in x.split(",")]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    return list(x)

biz_cuisines = (rest[[REST_ID_COL, CUISINE_COL]]
                .drop_duplicates(REST_ID_COL)
                .assign(cuisine=lambda d: d[CUISINE_COL].map(to_list))
                .explode("cuisine")
                .dropna(subset=["cuisine"]))
biz_cuisines["cuisine"] = biz_cuisines["cuisine"].str.strip()
biz_cuisines = biz_cuisines[(biz_cuisines["cuisine"] != "") &
                            ~biz_cuisines["cuisine"].str.lower().isin(GENERIC)]

top_cuisines = biz_cuisines["cuisine"].value_counts().head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_cuisines.values, y=top_cuisines.index, color="steelblue")
plt.title("Top 15 cuisines (by number of restaurants)")
plt.xlabel("Restaurants")
plt.show()

# MOST COMMON WORDS BY CUISINE

N_CUISINES = 6 # only provides graph for the top 6 cuisines, can edit for more graphs
review_cuisine = sample[[BIZ_ID_COL, TEXT_COL]].merge(
    biz_cuisines[[REST_ID_COL, "cuisine"]],
    left_on=BIZ_ID_COL, right_on=REST_ID_COL)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, cuisine in zip(axes.flat, top_cuisines.index[:N_CUISINES]):
    words, counts = zip(*top_words(review_cuisine.loc[review_cuisine["cuisine"] == cuisine, TEXT_COL]))
    sns.barplot(x=list(counts), y=list(words), ax=ax, color="steelblue")
    ax.set_title(cuisine)
    ax.set_xlabel("Count")
plt.tight_layout()
plt.show()
