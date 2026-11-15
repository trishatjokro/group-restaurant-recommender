import { router } from 'expo-router';
import { StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';

// Matches designs/welcome.png
export default function Welcome() {
  return (
    <SafeAreaView edges={['top', 'bottom']} style={styles.screen}>
      <View style={styles.header}>
        <Text style={styles.kicker}>welcome to your</Text>
        <Text style={styles.wordmark}>palate.</Text>
      </View>

      <View style={{ flex: 1 }} />

      <Text style={styles.title}>Good food.{'\n'}Better company.</Text>
      <Text style={styles.subtitle}>Welcome to palate. Your next favorite spot starts with you.</Text>

      <View style={styles.footer}>
        <PrimaryButton label="Let’s get started" onPress={() => router.push('/terms')} />
        <Text style={styles.caption}>A little about you. A lot to look forward to.</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: Colors.cream, paddingHorizontal: 28 },
  header: { alignItems: 'center', marginTop: 140 },
  kicker: { fontFamily: Fonts.bold, fontSize: 30, color: Colors.primaryDark, letterSpacing: -0.3 },
  wordmark: { fontFamily: Fonts.logo, fontSize: 64, lineHeight: 72, color: Colors.primary, letterSpacing: -1 },
  title: { fontFamily: Fonts.bold, fontSize: 34, lineHeight: 46, color: Colors.text },
  subtitle: { fontFamily: Fonts.regular, fontSize: 16, lineHeight: 23, color: Colors.muted, marginTop: 10 },
  footer: { marginTop: 72, marginBottom: 12 },
  caption: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, textAlign: 'center', marginTop: 12 },
});
