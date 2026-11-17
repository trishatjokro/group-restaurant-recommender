# Palate API contract

Start the server from the repo's top folder:
    .venv/bin/python -m uvicorn api.app:app --port 8000

## POST /recommend
Request:
```json
{"members": [
   {"name": "Trisha", "bubbles": ["Japanese", "Coffee & Tea"], "loves_text": "",
    "avoid": ["Loud"], "avoid_other": "", "loved_restaurants": ["Morimoto"], "prices": [1, 2]},
   {"name": "Connor", "skipped": true}
 ],
 "top_k": 5}
```
- bubbles = tapped cuisines/foods; loves_text = anything typed; prices: 1=$, 2=$$, 3=$$$ (empty = any)

Response:
```json
{"results": [
   {"name": "Hiroki", "categories": "Japanese, Sushi Bars", "price": "$$$", "stars": 4.5,
    "address": "...", "match_pct": 99, "best_fit_for": "Trisha",
    "member_fit": {"Trisha": 98, "Connor": 81},
    "reason": "Fits Trisha's picks: Japanese", "label": null}
 ]}
```
- match_pct / member_fit: 0–100 (how high it ranks among all restaurants for the group / each person)
- label: set when a result matches nobody's tapped cuisine, e.g.
  "Not a Japanese restaurant, but reviewers describe similar dishes"

## GET /search?q=mor   → {"names": ["Morimoto", ...]}   (for "Restaurants you love")
## GET /health         → {"ok": true}
