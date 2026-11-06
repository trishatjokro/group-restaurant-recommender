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

## 8. Which way of building the group vector works best? (Nov 6)
**Tried:** 5 ways to combine members into one group vector: average, weighted
(by how much each person told us), least-misery vector, anchored (average of the
top 20 restaurants everyone rates OK), anchored + normalized (each person's scores
ranked first).
**Test 1, 2,000 real Yelp friend pairs** (hide their most recent shared favorite):
average 0.9% hit@10 / 3.9% hit@50; weighted 1.1% / 3.8%; least misery = average
(identical for 2 people); anchored 0.4% / 2.4%; anchored+normalized 0.3% / 2.1%;
random 0.1% / 0.7%.
**Test 2, 300 mock 4-person groups** (does everyone get ≥1 result that fits?):
average 89% of members / 66% of groups with nobody left out; weighted same;
least misery 87% / 61%; anchored 80% / 46%; anchored+normalized 84% / 54%.
**Decision:** use the **average** (or weighted) group vector. The "smarter" vectors were
worse: anchoring pulls the group toward generic restaurants.
**Open problem:** in ~1 in 3 four-person groups someone is still left out; no single
vector fixed this.

## 9. Looking at real examples of group recommendations (Nov 6)
**Tried:** Ran the average group vector on example mock groups of 2, 3 and 4 and read the results.
**Found:** (a) When tastes overlap (3 people who all like sandwiches), results fit everyone.
(b) Midpoint problem confirmed: American/Italian + Korean/Japanese → Mexican and Middle
Eastern places neither person asked for. (c) A person who taps ONE bubble dominates
people who tap many: their vector is sharper, so a 4-person group got 5 Vietnamese places
for the one Vietnamese fan, and the Italian/Indian/pizza fan got nothing.
(d) Our automatic "does it fit?" check was too lenient (loose review keywords), so
experiment 8's "nobody left out" rates are too optimistic.
**Next:** make the fit check stricter; then test giving every person equal influence
regardless of how many bubbles they tap.
