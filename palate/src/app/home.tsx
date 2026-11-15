import { router } from 'expo-router';
import { useState } from 'react';
import { KeyboardAvoidingView, Platform, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BirdIcon, Blob, Duck, Star } from '@/components/palate/duck';
import { BottomBar, Logo, PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { useSession } from '@/lib/session';
import { resetOnboarding } from '@/lib/storage';

const MAX_PEOPLE = 10;

// Flowchart 2: group session creation.
export default function Home() {
  const { start } = useSession();
  const [sessionName, setSessionName] = useState('');
  const [names, setNames] = useState(['', '', '']);
  const [error, setError] = useState<string | null>(null);

  const setName = (i: number, value: string) => {
    setNames((n) => n.map((v, j) => (j === i ? value : v)));
    setError(null);
  };

  const submit = () => {
    const valid = names.map((n) => n.trim()).filter(Boolean);
    if (valid.length < 2) return setError('Add at least two people to start a group.');
    const lower = valid.map((n) => n.toLowerCase());
    if (new Set(lower).size !== lower.length) return setError('Each person needs a different name.');
    start(sessionName, valid);
    router.push('/handoff');
  };

  return (
    <KeyboardAvoidingView style={{ flex: 1, backgroundColor: Colors.white }} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
      <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={{ flexGrow: 1 }} bounces={false}>
        <View style={styles.hero}>
          <View style={styles.blob}>
            <Blob width={190} />
          </View>
          <SafeAreaView edges={['top']} style={{ paddingHorizontal: 28, paddingTop: 12 }}>
            {__DEV__ ? (
              // Dev-only: long-press the logo to replay onboarding (welcome + terms) for demos.
              <Pressable
                onLongPress={() => {
                  resetOnboarding();
                  router.replace('/welcome');
                }}
                delayLongPress={600}>
                <Logo />
              </Pressable>
            ) : (
              <Logo />
            )}
            <Text style={styles.heroTitle}>Let’s grab a bite!</Text>
          </SafeAreaView>
          <View style={styles.star}>
            <Star size={46} />
          </View>
          <View style={styles.duck}>
            <Duck width={260} />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.title}>Plan your next meal.</Text>
          <Text style={styles.subtitle}>Choose your crew to get started.</Text>

          <Text style={styles.label}>Session name</Text>
          <TextInput
            value={sessionName}
            onChangeText={setSessionName}
            placeholder="e.g. Dinner"
            placeholderTextColor={Colors.muted}
            style={styles.input}
            returnKeyType="next"
          />

          <View style={styles.peopleHeader}>
            <Text style={[styles.label, { marginTop: 0, marginBottom: 0 }]}>Who’s coming along?</Text>
            {names.length < MAX_PEOPLE ? (
              <Pressable accessibilityRole="button" onPress={() => setNames((n) => [...n, ''])} style={styles.addPerson}>
                <Text style={styles.addPersonText}>+ Add person</Text>
              </Pressable>
            ) : null}
          </View>

          {names.map((name, i) => (
            <View key={i} style={styles.personRow}>
              <BirdIcon />
              <TextInput
                value={name}
                onChangeText={(v) => setName(i, v)}
                placeholder={`Person ${i + 1} · optional`}
                placeholderTextColor={Colors.muted}
                style={styles.personInput}
                autoCapitalize="words"
                autoCorrect={false}
                maxLength={20}
              />
            </View>
          ))}

          {error ? <Text style={styles.error}>{error}</Text> : null}
        </View>
      </ScrollView>

      <BottomBar>
        <PrimaryButton label="Find a place to eat" onPress={submit} />
      </BottomBar>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  hero: { backgroundColor: Colors.cream, height: 360, overflow: 'hidden' },
  blob: { position: 'absolute', top: 0, right: 0 },
  heroTitle: { fontFamily: Fonts.bold, fontSize: 34, color: Colors.text, textAlign: 'center', marginTop: 34 },
  star: { position: 'absolute', top: 205, right: 70, transform: [{ rotate: '-12deg' }] },
  duck: { position: 'absolute', bottom: -70, left: -30 },
  card: {
    flex: 1,
    backgroundColor: Colors.white,
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    marginTop: -28,
    paddingHorizontal: 28,
    paddingTop: 32,
    paddingBottom: 24,
  },
  title: { fontFamily: Fonts.bold, fontSize: 32, color: Colors.text, textAlign: 'center' },
  subtitle: { fontFamily: Fonts.regular, fontSize: 16, color: Colors.muted, textAlign: 'center', marginTop: 10 },
  label: { fontFamily: Fonts.semibold, fontSize: 16, color: Colors.text, marginTop: 28, marginBottom: 10 },
  input: {
    backgroundColor: Colors.inputBg,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 14,
    paddingHorizontal: 18,
    paddingVertical: 15,
    fontFamily: Fonts.regular,
    fontSize: 16,
    color: Colors.text,
  },
  peopleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 28,
    marginBottom: 10,
  },
  addPerson: { backgroundColor: Colors.addPersonBg, borderRadius: 999, paddingHorizontal: 12, paddingVertical: 7 },
  addPersonText: { fontFamily: Fonts.semibold, fontSize: 13, color: Colors.accent },
  personRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: Colors.inputBg,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 14,
    paddingHorizontal: 16,
    marginBottom: 10,
  },
  personInput: { flex: 1, paddingVertical: 15, fontFamily: Fonts.regular, fontSize: 16, color: Colors.text },
  error: { fontFamily: Fonts.medium, fontSize: 14, color: Colors.error, marginTop: 4 },
});
