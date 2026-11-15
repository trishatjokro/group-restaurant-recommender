import { Redirect, router } from 'expo-router';
import { useMemo, useState } from 'react';
import { Pressable, RefreshControl, ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { BottomBar, GhostButton, Logo, PrimaryButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { goHome } from '@/lib/nav';
import { getRecommendations, joinWords, type Ranked } from '@/lib/recommend';
import { useSession } from '@/lib/session';

const PAGE = 5;

// Flowchart 5: show top 5, refresh for the next 5, "Why this match?", pick one.
export default function Results() {
  const { session, choose } = useSession();
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);

  const people = session?.participants;
  const recs = useMemo(() => (people ? getRecommendations(people, PAGE) : null), [people]);

  if (!session || !recs) return <Redirect href="/home" />;

  const pages = Math.max(1, Math.ceil(recs.ranked.length / PAGE));
  const shown = recs.ranked.slice((page % pages) * PAGE, (page % pages) * PAGE + PAGE);

  const nextPage = () => {
    setSelected(null);
    setPage((p) => p + 1);
  };

  const confirm = () => {
    if (!selected) return;
    choose(selected);
    router.replace('/completed');
  };

  return (
    <View style={{ flex: 1, backgroundColor: Colors.cream }}>
      <ScrollView
        contentContainerStyle={styles.scroll}
        refreshControl={<RefreshControl refreshing={false} onRefresh={nextPage} tintColor={Colors.primary} />}>
        <SafeAreaView edges={['top']}>
          <View style={styles.headerRow}>
            <Logo />
            <Pressable accessibilityRole="button" onPress={goHome} hitSlop={10}>
              <Text style={styles.startOver}>Start over</Text>
            </Pressable>
          </View>
          <Text style={styles.title}>
            {session.name} <Text style={{ color: Colors.primary }}>matches</Text>
          </Text>
          <Text style={styles.subtitle}>
            Top picks for {joinWords(session.participants.map((p) => p.name))}. Tap one you like.
          </Text>
          {recs.relaxed !== 'strict' ? (
            <Text style={styles.notice}>
              {recs.relaxed === 'relaxDiet'
                ? 'Not enough places matched everyone’s dietary needs, so we widened the search.'
                : 'Not enough places matched everyone’s budget and dietary needs, so we widened the search.'}
            </Text>
          ) : null}
        </SafeAreaView>

        {shown.map((item, i) => (
          <RestaurantCard
            key={item.restaurant.id}
            item={item}
            rank={(page % pages) * PAGE + i + 1}
            selected={selected === item.restaurant.id}
            onSelect={() => setSelected((s) => (s === item.restaurant.id ? null : item.restaurant.id))}
          />
        ))}

        {pages > 1 ? <GhostButton label="Show me 5 more ↻" onPress={nextPage} /> : null}
      </ScrollView>

      <BottomBar>
        <PrimaryButton label="It’s a food date!" disabled={!selected} onPress={confirm} />
      </BottomBar>
    </View>
  );
}

function RestaurantCard({
  item,
  rank,
  selected,
  onSelect,
}: {
  item: Ranked;
  rank: number;
  selected: boolean;
  onSelect: () => void;
}) {
  const r = item.restaurant;
  const meta = [`★ ${r.stars.toFixed(1)}`, r.price ? '$'.repeat(r.price) : null, `${r.reviews.toLocaleString()} reviews`]
    .filter(Boolean)
    .join('  ·  ');

  return (
    <Pressable
      accessibilityRole="radio"
      accessibilityState={{ selected }}
      onPress={onSelect}
      style={[styles.card, selected && styles.cardSelected]}>
      <View style={styles.cardTop}>
        <Text style={styles.rank}>{rank}</Text>
        <View style={{ flex: 1 }}>
          <Text style={styles.name}>{r.name}</Text>
          <Text style={styles.meta}>{meta}</Text>
        </View>
        <View style={styles.badge}>
          <Text style={styles.badgeText}>{Math.round(item.score * 100)}%</Text>
          <Text style={styles.badgeLabel}>match</Text>
        </View>
      </View>

      <View style={styles.tags}>
        {r.categories.slice(0, 3).map((c) => (
          <Text key={c} style={styles.tag}>
            {c}
          </Text>
        ))}
      </View>
      <Text style={styles.address}>{r.address}</Text>

      <Pressable
        accessibilityRole="button"
        hitSlop={8}
        onPress={() => router.push({ pathname: '/match/[id]', params: { id: r.id } })}
        style={{ alignSelf: 'flex-start', marginTop: 14 }}>
        <Text style={styles.why}>Why this match?</Text>
      </Pressable>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  scroll: { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 24 },
  headerRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 4 },
  startOver: { fontFamily: Fonts.semibold, fontSize: 14, color: Colors.accent },
  title: { fontFamily: Fonts.bold, fontSize: 34, color: Colors.text, marginTop: 24, paddingHorizontal: 4 },
  subtitle: { fontFamily: Fonts.regular, fontSize: 15, color: Colors.muted, marginTop: 8, marginBottom: 20, paddingHorizontal: 4 },
  notice: {
    fontFamily: Fonts.medium,
    fontSize: 13,
    color: Colors.accent,
    backgroundColor: Colors.addPersonBg,
    borderRadius: 12,
    padding: 12,
    marginBottom: 16,
    overflow: 'hidden',
  },
  card: {
    backgroundColor: Colors.white,
    borderRadius: 20,
    padding: 18,
    marginBottom: 14,
    borderWidth: 2,
    borderColor: Colors.white,
  },
  cardSelected: { borderColor: Colors.primary },
  cardTop: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  rank: { fontFamily: Fonts.logo, fontSize: 24, color: Colors.primary, width: 24, textAlign: 'center' },
  name: { fontFamily: Fonts.bold, fontSize: 18, color: Colors.text },
  meta: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, marginTop: 4 },
  badge: { alignItems: 'center', backgroundColor: Colors.addPersonBg, borderRadius: 12, paddingHorizontal: 10, paddingVertical: 6 },
  badgeText: { fontFamily: Fonts.bold, fontSize: 16, color: Colors.primaryDark },
  badgeLabel: { fontFamily: Fonts.medium, fontSize: 10, color: Colors.accent },
  tags: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 14, marginLeft: 36 },
  tag: {
    fontFamily: Fonts.medium,
    fontSize: 12,
    color: Colors.text,
    borderWidth: 1,
    borderColor: Colors.chipBorder,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 5,
    overflow: 'hidden',
  },
  address: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, marginTop: 10, marginLeft: 36 },
  why: { fontFamily: Fonts.semibold, fontSize: 14, color: Colors.accent, marginLeft: 36 },
});
