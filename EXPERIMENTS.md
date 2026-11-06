# Palate: Experiment Log

What we tried, what happened, and what we decided, in order.
All numbers are on Philadelphia (Yelp Open Dataset) unless noted.

---

## 1. Restaurant taste vectors from reviews (Oct 22)
**Tried:** Embed each open restaurant's reviews with SBERT (`all-MiniLM-L6-v2`, 384 numbers)
and average them into one vector per restaurant.
**Choices:** liked reviews only (low-star reviews are mostly about service, see EDA);
up to 50 most recent; restaurants need 15+ reviews and 3+ liked reviews.
**Result:** 7,021 restaurant vectors. Sanity check: Pat's → Geno's, Sonny's, Jim's (all
cheesesteaks) even though Yelp tags Pat's "Italian". Only 6% of reviews exceed the 256-token limit.
**Decision:** keep this as the restaurant side of the model.

## 2. Can the Yelp API's data replace the Open Dataset? (Oct 29)
**Tried:** Simulated the paid API's data (3 reviews × 160 characters per restaurant).
**Result:** "Similar restaurant" lists changed almost completely (<1 of 10 in common).
Same-cuisine neighbours dropped from 87% (full data) to 61% (random = 15%).
**Decision:** keep Philadelphia (full reviews) as the main demo; an LA version via the API
would be noticeably worse and is optional.

## 3. Taste profiles from questionnaire answers (Oct 27)
**Tried:** Turn loves / dislikes / price / diet into a person's vector (`taste_profile.py`).
**Result:** good matches (noodle lovers → noodle shops; Pat's → cheesesteak shops).
**Problems found:** (a) "dislikes" barely changed rankings, because every restaurant's score
dropped by about the same amount; (b) diet "prefer" was too strong (noodle lover + "prefer vegan" →
all-vegan places).
**Decision:** diets got "must have / prefer"; dislikes and "prefer" strength still need fixing.

## 4. Should restaurant categories go into the vectors? (Nov 4–5)
**Tried:** Add each restaurant's Yelp categories to its vector, at several weights (0–35%).
**First result (looked great):** tapping a cuisine bubble returned the right cuisine 82% → 97%.
**But:** that check was graded with the same tags we added (circular). Re-graded by what
**reviews** say, the gain was only 90% → 92%.
**Fair test (guessing game, 2,000 fresh users):** categories **lowered** accuracy:
hit@10 1.1% → 0.7–0.9%, with 95% confidence intervals below zero at 10–15% weight.
**Decision:** do **not** add categories to the vectors. Lesson: check whether a metric
is circular before trusting it.

## 5. Single words vs. descriptions for bubbles (Nov 5)
**Tried:** One word ("japanese") vs. a description ("Japanese food: sushi, ramen, udon").
**Result:** descriptions match better (88% → 92%). Tapping several bubbles at once:
joining them into one text lost picks (8% of mock users had all 5 results off);
embedding each description **separately** and averaging cut that to 3% and covered
the most picks (56% → 65%).
**Decision:** use separate bubble descriptions (not built yet).

## 6. "People who like X also like this" label (Nov 5)
**Tried:** Show non-matching results with "fans of X also love this", computed from real
users' likes.
**Result:** numbers were tiny (0.2–3%) and noisy (a liquor store, a skating rink and
Barnes & Noble ranked as "related").
**Decision:** use honest wording instead: "Not a ___ restaurant, but reviewers describe
similar dishes." Side finding: some non-restaurants are tagged "Restaurants" and should be
filtered out.

## 7. Combining a group into one group vector (Nov 6)
**Tried:** 4 methods in `group_vector.py`: average, weighted, least misery, anchored.
**Smoke test:** A = "spicy sichuan chinese", B = "creamy western, burgers". **All methods
recommended only Chinese places**, so B was ignored.
**Why:** some tastes match restaurants more strongly than others, so one person's scores dominate.
**Next:** test score normalization (rank each person's scores before combining) on real
friend pairs + mock 4-person groups.
