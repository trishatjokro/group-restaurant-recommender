// Web version of storage.ts: the browser's localStorage instead of expo-sqlite,
// which needs extra WebAssembly setup to bundle for web.

const ONBOARDED_KEY = 'palate.onboarded.v1';

export function hasOnboarded(): boolean {
  try {
    return globalThis.localStorage?.getItem(ONBOARDED_KEY) === 'true';
  } catch {
    return false;
  }
}

export function markOnboarded() {
  try {
    globalThis.localStorage?.setItem(ONBOARDED_KEY, 'true');
  } catch {
    // Non-fatal: the user will just see onboarding again next visit.
  }
}

// Dev/demo helper: forget onboarding so the welcome + terms flow shows again.
export function resetOnboarding() {
  try {
    globalThis.localStorage?.removeItem(ONBOARDED_KEY);
  } catch {
    // Non-fatal.
  }
}
