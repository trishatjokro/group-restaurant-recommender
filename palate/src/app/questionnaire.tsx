import { Redirect, router } from 'expo-router';
import { useState } from 'react';
import { Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BottomBar, Chip, Logo, PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import {
  AVOIDS,
  CUISINES,
  DIETS,
  MORE_CUISINES,
  MORE_SPECIFICS,
  NO_DIET,
  PRICES,
  SPECIFICS,
  type Option,
} from '@/lib/options';
import { useSession } from '@/lib/session';

// Flowchart 3: individual participant preferences. Every question is skippable.
export default function Questionnaire() {
  const { session, submitAnswers } = useSession();
  const [cuisines, setCuisines] = useState<string[]>([]);
  const [specifics, setSpecifics] = useState<string[]>([]);
  const [prices, setPrices] = useState<number[]>([]);
  const [diets, setDiets] = useState<string[]>([]);
  const [avoids, setAvoids] = useState<string[]>([]);
  const [submitted, setSubmitted] = useState(false);

  if (submitted) return null; // navigating away; session.current already moved on
  if (!session) return <Redirect href="/home" />;
  const { participants, current } = session;
  const person = participants[current];
  if (!person) return <Redirect href="/results" />;

  const total = participants.length;
  const completed = participants.filter((p) => p.done).length;
  const isLast = current === total - 1;

  const toggleDiet = (key: string) =>
    setDiets((d) => {
      if (key === NO_DIET) return d.includes(NO_DIET) ? [] : [NO_DIET];
      const without = d.filter((x) => x !== NO_DIET);
      return without.includes(key) ? without.filter((x) => x !== key) : [...without, key];
    });

  const done = () => {
    setSubmitted(true);
    submitAnswers({ cuisines, specifics, prices, diets, avoids });
    router.replace(isLast ? '/results' : '/handoff');
  };

  return (
    <View style={{ flex: 1, backgroundColor: Colors.cream }}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <SafeAreaView edges={['top']}>
          <Logo />
          <Text style={styles.title}>
            {person.name}’s <Text style={{ color: Colors.primary }}>Palate</Text>
          </Text>
          <Text style={styles.progressLabel}>
            Person {current + 1} of {total} · {completed} {completed === 1 ? 'person' : 'people'} completed
          </Text>
          <View style={styles.progress}>
            {participants.map((p, i) => (
              <View
                key={i}
                style={[
                  styles.segment,
                  {
                    backgroundColor: p.done ? Colors.progressDone : i === current ? Colors.progressCurrent : Colors.progressTodo,
                  },
                ]}
              />
            ))}
          </View>
        </SafeAreaView>

        <Section n="01" kicker="CUISINE" title="What are you craving?" hint="Choose the cuisines that sound good.">
          <ChipGroup options={CUISINES} more={MORE_CUISINES} selected={cuisines} onChange={setCuisines} />
        </Section>

        <Section n="02" kicker="SOMETHING SPECIFIC" title="Anything specific?" hint="Narrow it down to a dish or type of place.">
          <ChipGroup options={SPECIFICS} more={MORE_SPECIFICS} selected={specifics} onChange={setSpecifics} />
        </Section>

        <Section n="03" kicker="BUDGET" title="What’s your budget?" hint="Choose a price range.">
          <View style={styles.priceRow}>
            {PRICES.map((p) => {
              const on = prices.includes(p.key);
              return (
                <Pressable
                  key={p.key}
                  accessibilityRole="checkbox"
                  accessibilityState={{ checked: on }}
                  accessibilityLabel={`${p.key} dollar signs`}
                  onPress={() => setPrices((ps) => (on ? ps.filter((x) => x !== p.key) : [...ps, p.key]))}
                  style={[styles.price, on && { backgroundColor: Colors.primary }]}>
                  <Text style={[styles.priceText, on && { color: Colors.white }]}>{p.label}</Text>
                </Pressable>
              );
            })}
          </View>
        </Section>

        <Section n="04" kicker="DIETARY NEEDS" title="Any dietary needs?" hint="Select any that apply to you.">
          <View style={styles.chips}>
            {DIETS.map((d) => (
              <Chip key={d.key} label={d.label} selected={diets.includes(d.key)} onPress={() => toggleDiet(d.key)} />
            ))}
          </View>
        </Section>

        <Section n="05" kicker="AVOID" title="Anything you’d rather avoid?" hint="Tell us what to leave out.">
          <View style={styles.chips}>
            {AVOIDS.map((a) => (
              <Chip
                key={a.key}
                label={a.label}
                selected={avoids.includes(a.key)}
                onPress={() => setAvoids((v) => (v.includes(a.key) ? v.filter((x) => x !== a.key) : [...v, a.key]))}
              />
            ))}
          </View>
        </Section>
      </ScrollView>

      <BottomBar>
        <PrimaryButton label={isLast ? 'Find my matches' : 'I’m done'} onPress={done} />
      </BottomBar>
    </View>
  );
}

function Section({
  n,
  kicker,
  title,
  hint,
  children,
}: {
  n: string;
  kicker: string;
  title: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.section}>
      <Text style={styles.kicker}>
        {n}  {kicker}
      </Text>
      <Text style={styles.qTitle}>{title}</Text>
      <Text style={styles.hint}>{hint}</Text>
      {children}
    </View>
  );
}

function ChipGroup({
  options,
  more,
  selected,
  onChange,
}: {
  options: Option[];
  more: Option[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  // Keep "more" open if something hidden is already selected.
  const [expanded, setExpanded] = useState(false);
  const showMore = expanded || more.some((o) => selected.includes(o.key));
  const visible = showMore ? [...options, ...more] : options;
  const toggle = (key: string) => onChange(selected.includes(key) ? selected.filter((k) => k !== key) : [...selected, key]);

  return (
    <>
      <View style={styles.chips}>
        {visible.map((o) => (
          <Chip key={o.key} label={o.label} selected={selected.includes(o.key)} onPress={() => toggle(o.key)} />
        ))}
      </View>
      <Pressable accessibilityRole="button" onPress={() => setExpanded(!showMore)} hitSlop={8} style={{ marginTop: 16 }}>
        <Text style={styles.seeMore}>{showMore ? 'See less  –' : `See more (${more.length})  +`}</Text>
      </Pressable>
    </>
  );
}

const styles = StyleSheet.create({
  scroll: { paddingHorizontal: 24, paddingTop: 12, paddingBottom: 32 },
  title: { fontFamily: Fonts.bold, fontSize: 40, color: Colors.text, textAlign: 'center', marginTop: 28 },
  progressLabel: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, marginTop: 18 },
  progress: { flexDirection: 'row', gap: 8, marginTop: 10 },
  segment: { flex: 1, height: 6, borderRadius: 3 },
  section: { marginTop: 36 },
  kicker: { fontFamily: Fonts.semibold, fontSize: 12, letterSpacing: 0.4, color: Colors.accent },
  qTitle: { fontFamily: Fonts.bold, fontSize: 22, color: Colors.text, marginTop: 8 },
  hint: { fontFamily: Fonts.regular, fontSize: 15, color: Colors.muted, marginTop: 6, marginBottom: 16 },
  chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  seeMore: { fontFamily: Fonts.semibold, fontSize: 15, color: Colors.accent },
  priceRow: { flexDirection: 'row', gap: 10 },
  price: {
    flex: 1,
    backgroundColor: Colors.white,
    borderRadius: 12,
    paddingVertical: 18,
    alignItems: 'center',
  },
  priceText: { fontFamily: Fonts.bold, fontSize: 19, color: Colors.text },
});
