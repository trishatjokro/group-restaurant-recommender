"""
Combine several people's taste profiles into ONE group vector, which is then
compared with every restaurant's vector (cosine similarity).

Methods (all return a single vector):
  "average"       baseline: the mean of everyone's taste vectors
  "weighted"      mean, weighted per person (e.g. by how much they told us)
  "least_misery"  the vector whose LOWEST similarity to any member is as high
                  as possible, so nobody is left far away
  "anchored"      find real restaurants that work for everyone (each restaurant's
                  score = its lowest member score), then point the group vector
                  at the average of the best ones, so it can't land on a
                  "midpoint" cuisine nobody asked for

People who skipped (no likes) are left out. If everyone skipped, the group
vector points at Philly's most popular well-rated restaurants.
"""

import numpy as np

from model.taste_profile import allowed, score_restaurants

ANCHOR_TOP_K = 20        # restaurants averaged in the "anchored" method
MISERY_STEPS = 200       # iterations for the "least_misery" method
POPULAR_TOP_K = 50       # restaurants averaged when everyone skipped


def _normalize(v):
    return v / np.linalg.norm(v)


def _popular_vector(restaurants, V):
    """Fallback when nobody gave preferences: well-rated, much-reviewed places."""
    score = restaurants["stars"] * np.log1p(restaurants["n_reviews"])
    top = np.argsort(-score.values)[:POPULAR_TOP_K]
    return _normalize(V[top].mean(axis=0))


def _least_misery_vector(U):
    """Start at the average, then keep nudging toward whoever is furthest away."""
    g = _normalize(U.mean(axis=0))
    for step in range(1, MISERY_STEPS + 1):
        worst = U[np.argmin(U @ g)]
        g = _normalize(g + worst / step)
    return g


def group_vector(members, restaurants, method="average", weights=None):
    """members: list of taste profiles from build_taste_profile().
    restaurants: the restaurant_vectors table.
    Returns {"vector": group vector, "dislikes": group dislikes vector or None}."""
    V = np.vstack(restaurants["vector"].values)
    active = [i for i, m in enumerate(members) if m["likes"] is not None]

    dislikes = [m["dislikes"] for m in members if m["dislikes"] is not None]
    group_dislikes = _normalize(np.mean(dislikes, axis=0)) if dislikes else None

    if not active:
        return {"vector": _popular_vector(restaurants, V), "dislikes": group_dislikes}

    U = np.vstack([members[i]["likes"] for i in active])

    if method == "average":
        g = _normalize(U.mean(axis=0))

    elif method == "weighted":
        w = np.ones(len(members)) if weights is None else np.asarray(weights, dtype=float)
        w = w[active]
        g = _normalize((U * w[:, None]).sum(axis=0) / w.sum())

    elif method == "least_misery":
        g = _least_misery_vector(U)

    elif method == "anchored":
        # each member's score for every restaurant (includes their dislikes and diet prefs)
        scores = np.vstack([score_restaurants(members[i], restaurants) for i in active])
        group_score = scores.min(axis=0)                     # the least-happy member decides
        ok = np.ones(len(restaurants), dtype=bool)
        visited = set()
        for m in members:
            ok &= allowed(m, restaurants)                    # everyone's price / must-have diet
            visited |= m["visited"]
        ok &= ~restaurants["business_id"].isin(visited).values
        group_score = np.where(ok, group_score, -np.inf)
        top = np.argsort(-group_score)[:ANCHOR_TOP_K]
        g = _normalize(V[top].mean(axis=0))

    else:
        raise ValueError(f"unknown method: {method}")

    return {"vector": g, "dislikes": group_dislikes}
