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

## 10. Accuracy by group size, with bootstrap confidence intervals (Nov 6)
**Tried:** 400 random groups per size (2–10 people, anyone with anyone) from the 1,000 mock
users; top 5 results; STRICT fit check (a result fits a person only if its Yelp cuisine tag
matches one of their picks); 95% CIs from 1,000 bootstrap resamples.
**Result:** people with ≥1 fitting result: average 79% (size 2) → 65% (size 10);
least misery 79% → 60%; random restaurants 66% → 58%. Groups where nobody is left out:
average 67% → 13%; least misery 67% → 11%; random 52% → 6%.
Chart: `results/group_size_accuracy.png`
**Decision:** the average group vector is best at every size (clearly so for 5–9 people).
Performance falls as groups grow; for the demo's max of 4, ~3 in 4 people get a matching result.
**Caveats:** random strangers are a harsh test (real friends overlap more); random restaurants
already score well because common tags (sandwiches, pizza) are everywhere, so the model's
value is the ~13-point gap over random.

## 11. Giving every person equal influence (Nov 6)
**Problem:** people who tap one bubble have sharp vectors and dominate people who tap many
(experiment 9).
**Tried (same setup as #10, 2–10 people, strict fit, bootstrap CIs):**
current (all bubbles joined into one text) vs. A (each bubble's description embedded
separately, then averaged per person) vs. B (A, then each person "sharpened" toward their
own top-20 restaurants). Also measured how often one person gets ≥4 of 5 results to themselves.
**Result (groups of 2 / 3 / 4):** people served: current 79/79/76%, A 84/85/79%, B 81/83/76%.
Nobody left out: current 67/57/46%, A 73/69/52%, B 70/65/45%. One person takes over:
current 24/16/7%, A 19/10/5%, B 27/14/7%. For 7+ people all versions are about equal.
Chart: `results/equal_influence.png`
**Decision:** build each person's vector with A (separate bubble descriptions); the group
vector stays a simple average. B overcorrects and is dropped.

## 12. Can we stop people being left out? (Nov 6)
**Tried (person vectors built with A; same setup as #10–11):**
(2) reweighting: rebuild the one group vector, giving more weight to anyone who got no result
in their personal top 5%, up to 6 rounds; (4) coverage re-ranking: take the group vector's top
50, then pick 5 so everyone is covered where possible. Both use only the model's own signals
(never the cuisine tags used for grading).
**Result:** small gains. Nobody left out (2 / 4 / 10 people): average 73/52/12%,
reweighting 75/53/13%, coverage 75/53/18%. People served improved by 1–4 points.
Chart: `results/left_out_methods.png`
**Why small:** the model often thinks a person is covered when their actual cuisine picks
aren't, so the bottleneck is the person vectors, not how the group is combined.
**Decision:** keep the simple average group vector (best story; nothing beats it meaningfully
for ≤4 people). Coverage re-ranking is future work for big groups. Improve person vectors
(build option A into taste_profile.py) next.

### Why we chose the average group vector in the end
We tried six ways to combine a group into one vector (experiments 7, 8, 10, 11, 12):
average, weighted, least-misery vector, anchored, anchored + normalized, and reweighting
(plus coverage re-ranking after the vector). We chose the **plain average** because:
1. **It performed best or tied for best in every test.** On 2,000 real Yelp friend pairs it found
   the pair's shared favorite ~9× more often than random; the anchored methods were worse
   (0.3–0.4% vs 0.9–1.1% hit@10), and the least-misery vector was worse for groups of 4+
   (e.g. 75% vs 69% of people served at 5 people).
2. **The "smarter" methods didn't fix the real problem.** Reweighting and coverage re-ranking
   added only 1–2 points for the demo's group sizes (2–4), within the confidence intervals.
3. **The biggest improvement came from better person vectors, not a fancier combination.**
   Embedding each bubble separately (experiment 11) raised "nobody left out" by 6–12 points,
   far more than any aggregation method.
4. **It's simple to explain and defend:** "we combine everyone's taste vectors into one
   group vector by averaging, then compare it with every restaurant."
5. **Its known limitation is honest and measurable:** with very different tastes (e.g. Chinese +
   burgers) the average can land between them; for big groups, coverage re-ranking is the
   most promising fix (+6 points "nobody left out" at 10 people).

## 13. Built option A into taste_profile.py (Nov 7)
**Change:** `build_taste_profile()` now takes `bubbles=[...]` (tapped bubbles); each is embedded
from its own short description (24 in `BUBBLE_DESCRIPTIONS`) and averaged with typed text,
loved restaurants and diet preferences. Typed text still goes in `loves_text`.
**Why:** experiment 11 (biggest measured gain: +6–12 points "nobody left out" for 2–4 people).
**Checked:** old calls unchanged; unknown bubbles fall back to their own name; group_vector works.
**Noticed:** overlapping picks count twice (Italian + Pizza outweigh Vietnamese), which is
reasonable, but worth knowing.
