// On-device stand-in for the Python recommender (Flowcharts 4 & 5).
//
// model/taste_profile.py needs sentence-transformers, which can't run on a phone,
// so this mirrors its structure with Yelp-category matching instead of embeddings:
//   allowed()           -> isAllowed()       (price + dietary hard filters)
//   score_restaurants() -> scorePerson()     (likes up, dislikes down, diet bonus)
//   group vector        -> groupScore()      (blend of average and least-happy person)
// To use the real model later, put it behind an API and swap getRecommendations().

import data from '@/data/restaurants.json';
import { labelFor, NO_DIET } from '@/lib/options';
import type { Participant } from '@/lib/session';

export type Restaurant = {
  id: string;
  name: string;
  address: string;
  stars: number;
  reviews: number;
  price: number | null;
  categories: string[];
  lat: number;
  lng: number;
};

export const RESTAURANTS = data as Restaurant[];

const BUBBLE_CATEGORIES: Record<string, string[]> = {
  Italian: ['Italian'],
  Mediterranean: ['Mediterranean', 'Greek', 'Middle Eastern', 'Lebanese', 'Falafel', 'Turkish'],
  Mexican: ['Mexican', 'Tex-Mex', 'Tacos'],
  French: ['French', 'Creperies'],
  Japanese: ['Japanese', 'Sushi Bars', 'Ramen'],
  Thai: ['Thai'],
  Korean: ['Korean'],
  Indian: ['Indian', 'Pakistani'],
  Chinese: ['Chinese', 'Dim Sum', 'Szechuan', 'Cantonese', 'Noodles'],
  Vietnamese: ['Vietnamese'],
  'American (New)': ['American (New)'],
  'American (Traditional)': ['American (Traditional)'],
  Sandwiches: ['Sandwiches'],
  Pizza: ['Pizza'],
  Burgers: ['Burgers'],
  'Breakfast & Brunch': ['Breakfast & Brunch'],
  'Coffee & Tea': ['Coffee & Tea'],
  Seafood: ['Seafood'],
  Cafes: ['Cafes'],
  Salad: ['Salad'],
  Delis: ['Delis'],
  Chicken: ['Chicken Wings', 'Chicken Shop'],
  Cheesesteak: ['Cheesesteaks'],
  'Fast Food': ['Fast Food'],
};

const AVOID_CATEGORIES: Record<string, string[]> = {
  Seafood: ['Seafood'],
  'Fast food': ['Fast Food'],
  Spicy: ['Szechuan', 'Hot Pot', 'Thai', 'Indian'],
  Loud: ['Bars', 'Sports Bars', 'Nightlife', 'Dive Bars', 'Pubs'],
};
const LONG_WAIT_REVIEWS = 1500; // very popular spots tend to have lines

const DIET_CATEGORIES: Record<string, string[]> = {
  vegetarian: ['Vegetarian', 'Vegan'],
  vegan: ['Vegan'],
  halal: ['Halal'],
  'gluten-free': ['Gluten-Free'],
};

const AVOID_PENALTY = 0.25;

const hasAny = (r: Restaurant, cats: string[]) => cats.some((c) => r.categories.includes(c));
const likesOf = (p: Participant) => [...p.cuisines, ...p.specifics];
const dietsOf = (p: Participant) => p.diets.filter((d) => d !== NO_DIET);

function matchedLikes(p: Participant, r: Restaurant) {
  return likesOf(p).filter((b) => hasAny(r, BUBBLE_CATEGORIES[b] ?? [b]));
}

function matchedAvoids(p: Participant, r: Restaurant) {
  return p.avoids.filter((a) =>
    a === 'Long waits' ? r.reviews >= LONG_WAIT_REVIEWS : hasAny(r, AVOID_CATEGORIES[a] ?? [a])
  );
}

const fitsPrice = (p: Participant, r: Restaurant) => !p.prices.length || r.price == null || p.prices.includes(r.price);
const fitsDiet = (p: Participant, r: Restaurant) => dietsOf(p).every((d) => hasAny(r, DIET_CATEGORIES[d] ?? []));

type Strictness = 'strict' | 'relaxDiet' | 'relaxAll';

function isAllowed(p: Participant, r: Restaurant, strictness: Strictness) {
  if (strictness === 'relaxAll') return true;
  if (!fitsPrice(p, r)) return false;
  return strictness === 'relaxDiet' || fitsDiet(p, r);
}

/** 0..1 score for one person (higher = better). */
export function scorePerson(p: Participant, r: Restaurant) {
  const likes = likesOf(p);
  const matched = matchedLikes(p, r);
  let s = !likes.length ? 0.6 : matched.length ? 0.72 + 0.08 * Math.min(matched.length - 1, 2) : 0.3;
  s += ((r.stars - 3.5) / 1.5) * 0.12; // quality prior
  s += Math.log10(Math.max(r.reviews, 1)) * 0.01; // tie-break on popularity
  s -= AVOID_PENALTY * matchedAvoids(p, r).length;
  if (dietsOf(p).some((d) => hasAny(r, DIET_CATEGORIES[d] ?? []))) s += 0.05;
  return Math.max(0, Math.min(1, s));
}

/** Blend of average and least-happy person so nobody gets steamrolled. */
export function groupScore(people: Participant[], r: Restaurant) {
  const scores = people.map((p) => scorePerson(p, r));
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  return 0.7 * avg + 0.3 * Math.min(...scores);
}

export type Ranked = { restaurant: Restaurant; score: number };
export type Recommendations = { ranked: Ranked[]; relaxed: Strictness };

/** Full ranked list; show it 5 at a time. Relaxes filters if too few places fit. */
export function getRecommendations(people: Participant[], minResults = 5): Recommendations {
  let pool: Restaurant[] = [];
  let relaxed: Strictness = 'strict';
  for (const level of ['strict', 'relaxDiet', 'relaxAll'] as const) {
    relaxed = level;
    pool = RESTAURANTS.filter((r) => people.every((p) => isAllowed(p, r, level)));
    if (pool.length >= minResults) break;
  }
  const ranked = pool
    .map((restaurant) => ({ restaurant, score: groupScore(people, restaurant) }))
    .sort((a, b) => b.score - a.score);
  return { ranked, relaxed };
}

export function getRestaurant(id: string) {
  return RESTAURANTS.find((r) => r.id === id);
}

// --- "Why this match?" (Flowchart 5) ---

export type PersonMatch = { name: string; score: number; line: string };

export function explainMatch(people: Participant[], r: Restaurant) {
  const perPerson: PersonMatch[] = people.map((p) => {
    const likes = matchedLikes(p, r).map(labelFor);
    const avoids = matchedAvoids(p, r);
    const parts: string[] = [];
    if (likes.length) parts.push(`craving ${joinWords(likes)}`);
    else if (likesOf(p).length) parts.push('not their top craving, but well rated');
    else parts.push('open to anything');
    if (avoids.length) parts.push(`heads up: ${joinWords(avoids.map((a) => a.toLowerCase()))}`);
    if (p.prices.length && r.price && p.prices.includes(r.price)) parts.push('in budget');
    return { name: p.name, score: scorePerson(p, r), line: capitalize(parts.join(' · ')) };
  });

  // Group the shared reasons: "Serves Italian, picked by Alex and Sam."
  const byBubble = new Map<string, string[]>();
  for (const p of people) {
    for (const b of matchedLikes(p, r)) byBubble.set(b, [...(byBubble.get(b) ?? []), p.name]);
  }
  const reasons = [...byBubble.entries()]
    .sort((a, b) => b[1].length - a[1].length)
    .slice(0, 3)
    .map(([b, names]) => `${labelFor(b)}, picked by ${joinWords(names)}`);

  const budgets = people.filter((p) => p.prices.length);
  const priceNote =
    r.price && budgets.length && budgets.every((p) => p.prices.includes(r.price!))
      ? `Fits everyone's budget (${'$'.repeat(r.price)}).`
      : null;

  return { perPerson, reasons, priceNote };
}

export function joinWords(items: string[]) {
  if (items.length <= 1) return items[0] ?? '';
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
