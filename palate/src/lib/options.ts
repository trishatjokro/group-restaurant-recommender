// Questionnaire options. Keys match BUBBLE_DESCRIPTIONS / DIET_CATEGORIES in
// model/taste_profile.py so answers can be sent to the Python model unchanged.

export type Option<T extends string | number = string> = { key: T; label: string };

const opts = (...keys: string[]): Option[] => keys.map((key) => ({ key, label: key }));

export const CUISINES = opts('Italian', 'Mediterranean', 'Mexican', 'French', 'Japanese', 'Thai', 'Korean', 'Indian');
export const MORE_CUISINES: Option[] = [
  { key: 'Chinese', label: 'Chinese' },
  { key: 'Vietnamese', label: 'Vietnamese' },
  { key: 'American (New)', label: 'New American' },
  { key: 'American (Traditional)', label: 'Classic American' },
];

export const SPECIFICS = opts('Sandwiches', 'Pizza', 'Burgers', 'Breakfast & Brunch', 'Coffee & Tea', 'Seafood');
export const MORE_SPECIFICS = opts('Cafes', 'Salad', 'Delis', 'Chicken', 'Cheesesteak', 'Fast Food');

export const PRICES: Option<number>[] = [
  { key: 1, label: '$' },
  { key: 2, label: '$$' },
  { key: 3, label: '$$$' },
];

export const NO_DIET = 'none';
export const DIETS: Option[] = [
  { key: 'vegetarian', label: 'Vegetarian' },
  { key: 'vegan', label: 'Vegan' },
  { key: 'gluten-free', label: 'Gluten-free' },
  { key: 'halal', label: 'Halal' },
  { key: NO_DIET, label: 'None' },
];

export const AVOIDS = opts('Seafood', 'Long waits', 'Loud', 'Fast food', 'Spicy');

const ALL_LABELS = new Map<string, string>(
  [...CUISINES, ...MORE_CUISINES, ...SPECIFICS, ...MORE_SPECIFICS, ...DIETS, ...AVOIDS].map((o) => [
    o.key,
    o.label,
  ])
);
export const labelFor = (key: string) => ALL_LABELS.get(key) ?? key;
