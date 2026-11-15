import { Redirect } from 'expo-router';

import { hasOnboarded } from '@/lib/storage';

// Flowchart 1: "Onboarding complete?" -> homepage, otherwise welcome page.
export default function Index() {
  return <Redirect href={hasOnboarded() ? '/home' : '/welcome'} />;
}
