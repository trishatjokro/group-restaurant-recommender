import { router } from 'expo-router';

/** Back to a fresh home screen with nothing left on the stack. */
export function goHome() {
  if (router.canDismiss()) router.dismissAll();
  router.replace('/home');
}
