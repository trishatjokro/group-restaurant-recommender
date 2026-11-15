import { router } from 'expo-router';
import { useState } from 'react';
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { Path } from 'react-native-svg';

import { GhostButton, Logo, PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { markOnboarded } from '@/lib/storage';

type Doc = 'terms' | 'privacy';

// TODO: replace with the real legal copy.
const DOCS: Record<Doc, { title: string; body: string }> = {
  terms: {
    title: 'Terms of Service',
    body:
      'palate. suggests restaurants based on the preferences your group enters. Recommendations come from public review data and may be out of date. Check hours, prices and dietary information with the restaurant before you go.\n\nBy using palate. you agree not to misuse the app and understand it is provided as-is for planning purposes.',
  },
  privacy: {
    title: 'Privacy Policy',
    body:
      'Your group’s names and answers stay on this device for the current session and are cleared when you start over. We only store whether you have finished onboarding.\n\nWe don’t sell or share your data.',
  },
};

const DISABLED_BG = '#D8C9BF';
const SUMMARY_BG = '#F1E7E0';

// Matches designs/terms and conditions.png
export default function Terms() {
  const [opened, setOpened] = useState<Record<Doc, boolean>>({ terms: false, privacy: false });
  const [showing, setShowing] = useState<Doc | null>(null);
  const [checked, setChecked] = useState(false);
  const [error, setError] = useState(false);
  const allRead = opened.terms && opened.privacy;

  const agree = () => {
    if (!allRead) return setError(true);
    markOnboarded();
    router.replace('/home');
  };

  return (
    <SafeAreaView edges={['top', 'bottom']} style={styles.screen}>
      <ScrollView contentContainerStyle={styles.body} bounces={false}>
        <Logo />
        <View style={styles.shield}>
          <ShieldCheck />
        </View>
        <Text style={styles.title}>One more thing!</Text>
        <Text style={styles.subtitle}>
          Before we find your next great meal, take a moment to review our terms and privacy information.
        </Text>

        <View style={styles.card}>
          {(Object.keys(DOCS) as Doc[]).map((doc) => (
            <Pressable
              key={doc}
              accessibilityRole="button"
              accessibilityHint={opened[doc] ? 'Already opened' : undefined}
              onPress={() => setShowing(doc)}
              style={({ pressed }) => [styles.row, pressed && { opacity: 0.6 }]}>
              <Text style={styles.rowText}>{DOCS[doc].title}</Text>
              <Arrow />
            </Pressable>
          ))}
        </View>

        <View style={styles.summary}>
          <Text style={styles.summaryTitle}>Terms & privacy summary</Text>
          <Text style={styles.summaryBody}>
            Your group’s names and answers stay on this device and are only used to find restaurants you’ll all enjoy.
            We don’t sell or share your data.
          </Text>
        </View>

        {error && !allRead ? (
          <Text style={styles.error}>
            Please open both the Terms of Service and the Privacy Policy before continuing.
          </Text>
        ) : null}

        <View style={{ flex: 1, minHeight: 32 }} />

        <Pressable
          accessibilityRole="checkbox"
          accessibilityState={{ checked }}
          onPress={() => setChecked((c) => !c)}
          style={styles.checkRow}>
          <View style={[styles.checkbox, checked && styles.checkboxOn]}>
            {checked ? (
              <Svg width={14} height={14} viewBox="0 0 24 24">
                <Path d="M5 12.5l4.5 4.5L19 7.5" stroke={Colors.white} strokeWidth={3} fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </Svg>
            ) : null}
          </View>
          <Text style={styles.checkText}>I agree to the Terms of Service and acknowledge the Privacy Policy.</Text>
        </Pressable>

        <View style={styles.footer}>
          <PrimaryButton
            label="Agree & get started"
            onPress={agree}
            disabled={!checked}
            style={!checked && { backgroundColor: DISABLED_BG, opacity: 1 }}
          />
          <Text style={[styles.caption, checked && { opacity: 0 }]}>Select the checkbox to continue.</Text>
        </View>
      </ScrollView>

      <Modal visible={showing !== null} animationType="slide" presentationStyle="pageSheet" onRequestClose={() => setShowing(null)}>
        {showing ? (
          <View style={styles.sheet}>
            <Text style={styles.sheetTitle}>{DOCS[showing].title}</Text>
            <ScrollView style={{ flex: 1 }}>
              <Text style={styles.sheetBody}>{DOCS[showing].body}</Text>
            </ScrollView>
            <GhostButton
              label="Close"
              onPress={() => {
                setOpened((o) => ({ ...o, [showing]: true }));
                setShowing(null);
              }}
            />
          </View>
        ) : null}
      </Modal>
    </SafeAreaView>
  );
}

function ShieldCheck() {
  return (
    <Svg width={60} height={60} viewBox="0 0 60 60">
      <Path
        d="M30 4L54 12v16c0 15-10 24-24 29C16 52 6 43 6 28V12L30 4z"
        stroke={Colors.primary}
        strokeWidth={2.5}
        fill="none"
        strokeLinejoin="round"
      />
      <Path d="M20 30l7 7 13-14" stroke={Colors.primary} strokeWidth={2.5} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

function Arrow() {
  return (
    <Svg width={28} height={20} viewBox="0 0 28 20">
      <Path d="M2 10h23M16 2l9 8-9 8" stroke={Colors.accent} strokeWidth={2} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </Svg>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: Colors.cream },
  body: { flexGrow: 1, paddingHorizontal: 32, paddingTop: 16 },
  shield: { marginTop: 40 },
  title: { fontFamily: Fonts.bold, fontSize: 34, color: Colors.text, marginTop: 28 },
  subtitle: { fontFamily: Fonts.regular, fontSize: 16, lineHeight: 23, color: Colors.muted, marginTop: 12 },
  card: {
    backgroundColor: Colors.white,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 20,
    paddingVertical: 6,
    marginTop: 24,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 20,
  },
  rowText: { fontFamily: Fonts.semibold, fontSize: 17, color: Colors.text },
  summary: { backgroundColor: SUMMARY_BG, borderRadius: 16, padding: 20, marginTop: 26 },
  summaryTitle: { fontFamily: Fonts.semibold, fontSize: 16, color: Colors.text },
  summaryBody: { fontFamily: Fonts.regular, fontSize: 14, lineHeight: 20, color: Colors.muted, marginTop: 8 },
  error: { fontFamily: Fonts.regular, fontSize: 14, lineHeight: 20, color: Colors.primaryDark, marginTop: 16, paddingHorizontal: 14 },
  checkRow: { flexDirection: 'row', alignItems: 'flex-start', gap: 14 },
  checkbox: {
    width: 28,
    height: 28,
    borderRadius: 7,
    borderWidth: 1.5,
    borderColor: Colors.muted,
    backgroundColor: Colors.white,
    alignItems: 'center',
    justifyContent: 'center',
  },
  checkboxOn: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  checkText: { flex: 1, fontFamily: Fonts.regular, fontSize: 15, lineHeight: 22, color: Colors.text },
  footer: { marginTop: 36, marginBottom: 12 },
  caption: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, textAlign: 'center', marginTop: 12 },
  sheet: { flex: 1, backgroundColor: Colors.white, padding: 28, paddingTop: 36 },
  sheetTitle: { fontFamily: Fonts.bold, fontSize: 26, color: Colors.text, marginBottom: 16 },
  sheetBody: { fontFamily: Fonts.regular, fontSize: 16, lineHeight: 24, color: Colors.text },
});
