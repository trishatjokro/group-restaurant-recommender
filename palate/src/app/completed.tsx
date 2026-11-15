import { Redirect } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Duck, Popper } from '@/components/palate/duck';
import { OutlineButton } from '@/components/palate/ui';
import { Colors, Fonts } from '@/constants/palette';
import { goHome } from '@/lib/nav';
import { getRestaurant } from '@/lib/recommend';
import { useSession } from '@/lib/session';

// "Everyone has chosen" confirmation screen.
export default function Completed() {
  const { session } = useSession();
  const r = session?.chosenId ? getRestaurant(session.chosenId) : undefined;
  if (!session || !r) return <Redirect href="/home" />;

  const count = session.participants.length;

  return (
    <ScrollView style={{ flex: 1, backgroundColor: Colors.cream }} contentContainerStyle={{ flexGrow: 1 }}>
      <SafeAreaView edges={['top', 'bottom']} style={styles.body}>
        <Text style={styles.title}>It’s a food date!</Text>
        <Text style={styles.subtitle}>Your crew picked a place.</Text>

        {/* Placeholder until the dataset has photos. */}
        <View style={styles.photo}>
          <Text style={styles.photoName}>{r.name}</Text>
          <Text style={styles.photoCaption}>{r.address}</Text>
        </View>

        <Text style={styles.name}>{r.name}</Text>
        <Text style={styles.meta}>
          {session.name} · {count} friends
        </Text>
        <Text style={styles.categories}>{r.categories.slice(0, 3).join(' · ')}</Text>

        <View style={{ marginTop: 28 }}>
          <OutlineButton label="Back to home" onPress={goHome} />
        </View>

        <View style={styles.duckArea}>
          <View style={styles.duckWrap}>
            <View style={styles.popper}>
              <Popper />
            </View>
            <Duck width={160} eye="wink" />
          </View>
        </View>
      </SafeAreaView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  body: { flex: 1, paddingHorizontal: 24, paddingTop: 72 },
  title: { fontFamily: Fonts.bold, fontSize: 30, color: Colors.text, textAlign: 'center' },
  subtitle: { fontFamily: Fonts.regular, fontSize: 16, color: Colors.muted, textAlign: 'center', marginTop: 12 },
  photo: {
    height: 240,
    borderRadius: 24,
    backgroundColor: '#F6E4DD',
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 24,
    marginTop: 28,
  },
  photoName: { fontFamily: Fonts.semibold, fontSize: 21, color: '#A0442F', textAlign: 'center' },
  photoCaption: { fontFamily: Fonts.regular, fontSize: 13, color: Colors.muted, marginTop: 8, textAlign: 'center' },
  name: { fontFamily: Fonts.bold, fontSize: 25, color: Colors.text, textAlign: 'center', marginTop: 28 },
  meta: { fontFamily: Fonts.regular, fontSize: 16, color: Colors.muted, textAlign: 'center', marginTop: 10 },
  categories: { fontFamily: Fonts.regular, fontSize: 14, color: Colors.muted, textAlign: 'center', marginTop: 8 },
  duckArea: { flex: 1, justifyContent: 'flex-end', marginTop: 24 },
  duckWrap: { alignSelf: 'flex-start', paddingLeft: 34, paddingTop: 8 },
  popper: { position: 'absolute', left: 0, top: 0 },
});
