# Palate: group restaurant recommender

🏆 **Winner of Best Overall at the DataSC project showcase (November 2026), out of 9 groups.**

DataSC project. Palate recommends restaurants a whole group will be happy with.
It turns Yelp reviews and each person's preferences into vectors with a
sentence-transformer (SBERT), scores every restaurant for each member, and combines
those scores into one group ranking.

**City:** Philadelphia (7,021 recommendable restaurants, 1.15M cleaned reviews)
**Team:** Trisha Tjokrosapoetro (lead), Anshika Jain, Gisella Mai, Connor Mao, Evelyn Shen

## How it works

1. **Restaurants:** each restaurant's liked reviews are embedded with SBERT
   (`all-MiniLM-L6-v2`, 384-D) and averaged into one restaurant vector.
2. **People:** each member's answers (cuisines, specific foods, loved restaurants)
   are embedded the same way into a taste profile. "Avoid" answers form a
   dislikes vector. Budget and dietary needs are hard filters, not part of the vector.
3. **Individual scores:** for each person, every restaurant is scored with a 50/50 mix
   of an SVM trained on that person's likes and dislikes and plain cosine similarity
   (`model/hybrid_model.py`).
4. **Group score (consensus):** members' scores are combined as
   average − 0.5 × disagreement, so places the group disagrees on are pushed down
   (`model/group_recommender.py`).
5. **Results:** everyone's budget and diet filters apply, one location per chain,
   top 5 shown in the app.

## Repo structure

```
├── make_sample.py, make_cities.py, pipeline.py, make_users.py   # data prep (see "Getting the data")
├── data/
│   ├── clean/                 # cleaning functions (checks, text, duplicates, filters)
│   ├── mock/mock_users.json   # 1,000 simulated users for testing
│   ├── output/                # cleaned 50k-review SAMPLE from October (not the full city data)
│   └── cities/philadelphia/   # full city data (mostly NOT on GitHub; see below)
├── EDA/                       # exploratory analysis scripts + charts
├── model/
│   ├── embed_restaurants.py   # one vector per restaurant
│   ├── taste_profile.py       # one taste profile per person + filters
│   ├── make_mock_profiles.py  # builds data/mock/mock_users.json
│   ├── group_vector.py        # baselines: average group vector (+ least misery)
│   ├── svm_recommend.py       # Anshika: per-person SVM (+ evaluate_svm.py, tune_svm.py)
│   ├── group_recommender.py   # Evelyn: average / least misery / consensus on scores
│   └── hybrid_model.py        # Evelyn: SVM + cosine per person → consensus (+ hybrid_test.py)
├── evaluation/
│   ├── compare_models.py      # all models on the same tests + paired statistical tests
│   └── compare_models2.py     # updated version using the current hybrid
├── api/                       # FastAPI server the app calls (see api/CONTRACT.md)
├── palate/                    # Connor's Expo / React Native app
├── results/                   # charts and result tables
└── EXPERIMENTS.md             # log of what we tried, what worked, what didn't
```

## Setup

1. Clone the repo and create a Python environment (Python 3.10–3.12).
2. Install packages:
   `pip install pandas pyarrow langdetect matplotlib seaborn scipy scikit-learn folium sentence-transformers umap-learn plotly fastapi uvicorn`
3. **Run everything from the repo's top folder**, e.g. `python -m evaluation.compare_models2`.
   In PyCharm: Edit Configurations → Edit configuration templates → Python →
   Working directory = the repo folder.

## Getting the data

The full city data is too big for GitHub (`reviews_clean.parquet` alone is 465 MB).
Put these files in **`data/cities/philadelphia/`**:

| File | Size | On GitHub? |
|---|---|---|
| `restaurant_vectors.parquet` | 18 MB | no: Drive or rebuild |
| `reviews_clean.parquet` | 465 MB | no: Drive or rebuild |
| `restaurants_clean.parquet` | 3 MB | no: Drive or rebuild |
| `friend_pairs.parquet` | 22 MB | **yes** |

> ⚠️ **Data location:** some newer scripts (`svm_recommend.py`, `hybrid_test.py`,
> `compare_models.py`, `compare_models2.py`) read from `data/output/` instead.
> `data/output/` on GitHub is only the October **sample**. Results from the sample
> are NOT comparable to results on the full data. Point those paths at
> `data/cities/philadelphia/` (or copy the full files into `data/output/`) before running.

To rebuild from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)
(edit the `RAW =` path at the top of each script):

| Step | Command | Output |
|---|---|---|
| 1. Split by city | `python make_cities.py` | `reviews.parquet`, `restaurants.parquet` |
| 2. Clean a city | set `CITY` in `pipeline.py`, then `python pipeline.py` | `reviews_clean.parquet`, `restaurants_clean.parquet` |
| 3. Users + friends | `python make_users.py` | `users.parquet`, `friend_pairs.parquet` |
| 4. Embed restaurants | `python model/embed_restaurants.py` | `restaurant_vectors.parquet` |

## Running the app

1. Start the API (loads the model once, ~10 s):
   `python -m uvicorn api.app:app --port 8000`
2. In another terminal: `cd palate && npm install && npx expo start --web`
3. If the API isn't running, the app falls back to simpler on-device matching.

## Evaluation

`python -m evaluation.compare_models2` (~15 min) runs every model on the same tests:

- **Test A: real Yelp friend pairs (5,000 pairs).** Hide a restaurant both friends
  liked, build their profiles from the rest of their history, and check whether it
  ranks in the top 10 of 7,021 (**hit@10**).
- **Test B: simulated groups (2,700 groups of 2–10 mock users).** A member is
  **served** if at least one of the top 5 matches a cuisine they picked (graded with
  Yelp category tags the models never see). Main metric: **nobody left out**, the
  share of groups where every member is served.
- **Statistics:** all tests are paired (same pairs and groups for every model):
  McNemar for yes/no metrics, Wilcoxon signed-rank for ranks and % served, paired
  bootstrap 95% CIs (2,000 resamples), Holm correction across all comparisons,
  and 5 re-runs with fresh samples.

**Results on the full Philadelphia data (Nov 18):**

| Model | Friend pairs hit@10 | Served | Nobody left out |
|---|---|---|---|
| Average (baseline) | 1.0% | 76.0% | 38.4% |
| Least misery (baseline) | 0.9% | 67.3% | 27.3% |
| SVM + average | 0.8% | 76.6% | 41.4% |
| Consensus | 1.0% | 74.7% | 38.3% |
| **Hybrid (SVM + cosine → consensus)** | not run | 76.4% | **41.4%** |
| Popularity (no taste) | 18.2% | 64.1% | 23.6% |
| Random | 0.1% | 63.4% | 21.0% |

- The **hybrid beats the average baseline** on nobody left out (+3.1 pts, Holm p = 0.006,
  better in 5/5 re-runs) and ties SVM + average.
- On real friend pairs, all taste models tie (~1%), while popularity wins (18%): friends
  often go to well-known places together. See Limitations in the presentation.

## Key decisions

- Only recommend **open** restaurants with **≥15 reviews**
- A review is **"liked"** if it's above that user's own average rating
- Restaurant vectors use **liked reviews only** (up to the 50 most recent)
- Budget and "must" diets are **hard filters**; "avoid" is soft (pushes places down)
- Results show **one location per chain**
- Full history of experiments: `EXPERIMENTS.md`
