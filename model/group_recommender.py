# 1. score each eligible restaurant separately for every group member using their individual taste profile.
# 2. combine those individual scores using three aggregation methods: average preference, least misery, and consensus.
# 3. use the consensus score to reward restaurants the group generally likes while penalizing restaurants with high disagreement between members.

import numpy as np 
from taste_profile import score_restaurants, allowed 

def group_scores(taste_profiles, restaurants): 
    """
    score every restaurant for the whole group. 
    returns: average_scores, least_misery_scores, consensus_scores, eligible
    """
    all_scores = [] 
    all_allowed = [] 
    
    # score restaurants separately for every person 
    for profile in taste_profiles: 
        scores = score_restaurants(profile, restaurants)
        eligible = allowed(profile, restaurants)

        all_scores.append(scores)
        all_allowed.append(eligible)

    # shape: 
    # number of users x number of restaurants 
    all_scores = np.vstack(all_scores)
    all_allowed = np.vstack(all_allowed)

    # restaurant must satisfy every person's hard constraints 
    eligible = all_allowed.all(axis = 0)

    # baseline 1: average preference 
    average_scores = all_scores.mean(axis = 0)

    # baseline 2: least misery 
    least_misery_scores = all_scores.min(axis = 0)

    # consensus model 
    relevance = all_scores.mean(axis = 0)
    disagreement = all_scores.std(axis = 0)

    consensus_scores = relevance - disagreement 

    return (
        average_scores, 
        least_misery_scores, 
        consensus_scores, 
        eligible, 
    )

# return top restaurants
def recommend_group(taste_profiles, restaurants, top_n = 5):
    (
        average_scores,
        least_misery_scores, 
        consensus_scores, 
        eligible, 
    ) = group_scores(taste_profiles, restaurants)

    results = restaurants.copy() 

    results["average_score"] = average_scores
    results["least_misery_score"] = least_misery_scores
    results["consensus_score"] = consensus_scores

    # remove restaurants that violate someone's hard constraint 
    results = results[eligible].copy() 

    # highest consensus score first 
    results = results.sort_values(
        "consensus_score", 
        ascending = False 
    )

    return results.head(top_n)