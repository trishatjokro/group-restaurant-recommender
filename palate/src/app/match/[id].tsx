import { router, useLocalSearchParams } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { GhostButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { explainMatch, getApiExtras, getRestaurantAny, groupScore } from '@/lib/recommend'; // CHANGED (Trisha): API-aware lookups
import { useSession } from '@/lib/session';

// "Why this Match?" sheet (Flowchart 5).
export default function WhyThisMatch() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const { session } = useSession();
  const r = getRestaurantAny(id); // CHANGED (Trisha): also finds restaurants returned by the model
  const extras = getApiExtras(id); // ADDED (Trisha): the model's group % and per-person %, if available

  if (!session || !r) {
    return (
      <View style={styles.sheet}>
        <Text style={styles.body}>This restaurant isn’t available anymore.</Text>
        <GhostButton label="Close" onPress={() => router.back()} />
      </View>
    );
  }

  const people = session.participants;
  const { perPerson, reasons, priceNote } = explainMatch(people, r);

  return (
    <ScrollView style={{ backgroundColor: Colors.white }} contentContainerStyle={styles.sheet}>
      <Text style={styles.kicker}>WHY THIS MATCH?</Text>
      <Text style={styles.title}>{r.name}</Text>
      {/* CHANGED (Trisha): use the model's % when the result came from the API */}
      <Text style={styles.group}>{extras ? extras.matchPct : Math.round(groupScore(people, r) * 100)}% group match</Text>
      {/* ADDED (Trisha): honest note when the place isn't the cuisine anyone tapped */}
      {extras?.label ? <Text style={[styles.body, { marginTop: 8 }]}>{extras.label}</Text> : null}

      {reasons.length || priceNote ? (
        <View style={styles.reasons}>
          {reasons.map((line) => (
            <Text key={line} style={styles.body}>
              • {line}
            </Text>
          ))}
          {priceNote ? <Text style={styles.body}>• {priceNote}</Text> : null}
        </View>
      ) : (
        <Text style={[styles.body, { marginTop: 16 }]}>
          A highly rated spot ({r.stars.toFixed(1)}★) that works for everyone’s budget and needs.
        </Text>
      )}

      <Text style={styles.section}>How it fits each person</Text>
      {perPerson.map((p) => {
        // CHANGED (Trisha): per-person % from the model when available, else on-device score
        const pct = extras?.memberFit[p.name] ?? Math.round(p.score * 100);
        return (
        <View key={p.name} style={styles.person}>
          <View style={styles.personTop}>
            <Text style={styles.personName}>{p.name}</Text>
            <Text style={styles.personScore}>{pct}%</Text>
          </View>
          <View style={styles.track}>
            <View style={[styles.fill, { width: `${pct}%` }]} />
          </View>
          <Text style={styles.personLine}>{p.line}</Text>
        </View>
        );
      })}

      <GhostButton label="Close" onPress={() => router.back()} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  sheet: { padding: 28, paddingTop: 32, backgroundColor: Colors.white },
  kicker: { fontFamily: Fonts.semibold, fontSize: 12, letterSpacing: 0.4, color: Colors.accent },
  title: { fontFamily: Fonts.bold, fontSize: 26, color: Colors.text, marginTop: 6 },
  group: { fontFamily: Fonts.semibold, fontSize: 15, color: Colors.primary, marginTop: 6 },
  reasons: { marginTop: 16, gap: 6 },
  body: { fontFamily: Fonts.regular, fontSize: 15, lineHeight: 22, color: Colors.text },
  section: { fontFamily: Fonts.bold, fontSize: 17, color: Colors.text, marginTop: 28, marginBottom: 6 },
  person: { paddingVertical: 12, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: Colors.border },
  personTop: { flexDirection: 'row', justifyContent: 'space-between' },
  personName: { fontFamily: Fonts.semibold, fontSize: 15, color: Colors.text },
  personScore: { fontFamily: Fonts.bold, fontSize: 15, color: Colors.primaryDark },
  track: { height: 6, borderRadius: 3, backgroundColor: Colors.progressTodo, marginTop: 8, overflow: 'hidden' },
  fill: { height: 6, borderRadius: 3, backgroundColor: Colors.primary },
  personLine: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, marginTop: 8 },
});
