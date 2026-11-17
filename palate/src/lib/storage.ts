import Storage from 'expo-sqlite/kv-store';

const ONBOARDED_KEY = 'palate.onboarded.v1';

export function hasOnboarded(): boolean {
  try {
    return Storage.getItemSync(ONBOARDED_KEY) === 'true';
  } catch {
    return false;
  }
}

export function markOnboarded() {
  try {
    Storage.setItemSync(ONBOARDED_KEY, 'true');
  } catch {
    // Non-fatal: the user will just see onboarding again next launch.
  }
}

// Dev/demo helper: forget onboarding so the welcome + terms flow shows again.
export function resetOnboarding() {
  try {
    Storage.removeItemSync(ONBOARDED_KEY);
  } catch {
    // Non-fatal.
  }
}
