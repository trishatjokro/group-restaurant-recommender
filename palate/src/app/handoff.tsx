import { Redirect, router } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Duck } from '@/components/palate/duck';
import { BottomBar, Logo, PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { useSession } from '@/lib/session';

// "Pass the phone to N" between participants.
export default function Handoff() {
  const { session } = useSession();
  if (!session) return <Redirect href="/home" />;

  const total = session.participants.length;
  const person = session.participants[session.current];
  if (!person) return <Redirect href="/results" />;

  return (
    <View style={{ flex: 1, backgroundColor: Colors.cream }}>
      <SafeAreaView edges={['top']} style={styles.body}>
        <Logo />
        <View style={styles.art}>
          <View style={styles.outer} />
          <View style={styles.inner} />
          <View style={[styles.dot, { width: 18, height: 18, backgroundColor: Colors.accent, top: '12%', left: '8%' }]} />
          <View style={[styles.dot, { width: 100, height: 100, backgroundColor: Colors.sage, top: '14%', right: '2%' }]} />
          <View style={[styles.dot, { width: 84, height: 84, backgroundColor: Colors.blush, bottom: '22%', left: '2%' }]} />
          <View style={[styles.dot, { width: 14, height: 14, backgroundColor: Colors.olive, bottom: '34%', right: '6%' }]} />
          <Duck width={250} eye="wink" />
        </View>

        <Text style={styles.kicker}>
          PERSON {session.current + 1} OF {total}
        </Text>
        <Text style={styles.title}>Pass the phone{'\n'}to {person.name}</Text>
        <Text style={styles.subtitle}>It’s time to choose your preferences!</Text>
      </SafeAreaView>

      <BottomBar>
        <PrimaryButton label={`I’m ${person.name} — let’s go`} onPress={() => router.replace('/questionnaire')} />
      </BottomBar>
    </View>
  );
}

const styles = StyleSheet.create({
  body: { flex: 1, paddingHorizontal: 28, paddingTop: 12 },
  art: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  outer: { position: 'absolute', width: 340, height: 340, borderRadius: 170, backgroundColor: Colors.sand },
  inner: { position: 'absolute', width: 260, height: 220, borderRadius: 130, backgroundColor: Colors.ivory },
  dot: { position: 'absolute', borderRadius: 999 },
  kicker: { fontFamily: Fonts.semibold, fontSize: 13, letterSpacing: 0.6, color: Colors.muted, textAlign: 'center' },
  title: { fontFamily: Fonts.bold, fontSize: 36, lineHeight: 42, color: Colors.text, textAlign: 'center', marginTop: 10 },
  subtitle: { fontFamily: Fonts.regular, fontSize: 16, color: Colors.muted, textAlign: 'center', marginTop: 18, marginBottom: 28 },
});
