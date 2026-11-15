import type { ReactNode } from 'react';
import { Pressable, StyleSheet, Text, View, type StyleProp, type TextStyle, type ViewStyle } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { Colors, Fonts } from '@/constants/palette';

export function Logo({ style }: { style?: StyleProp<TextStyle> }) {
  return <Text style={[styles.logo, style]}>palate.</Text>;
}

export function PrimaryButton({
  label,
  onPress,
  disabled,
  arrow = true,
  style,
}: {
  label: string;
  onPress: () => void;
  disabled?: boolean;
  arrow?: boolean;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled }}
      onPress={onPress}
      disabled={disabled}
      style={({ pressed }) => [
        styles.button,
        pressed && { backgroundColor: Colors.primaryDark },
        disabled && { opacity: 0.45 },
        style,
      ]}>
      <Text style={styles.buttonText}>
        {label}
        {arrow ? '  →' : ''}
      </Text>
    </Pressable>
  );
}

export function OutlineButton({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      onPress={onPress}
      style={({ pressed }) => [styles.outline, pressed && { backgroundColor: Colors.inputBg }]}>
      <Text style={styles.outlineText}>{label}  →</Text>
    </Pressable>
  );
}

export function GhostButton({ label, onPress }: { label: string; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} style={({ pressed }) => [styles.ghost, pressed && { opacity: 0.6 }]}>
      <Text style={styles.ghostText}>{label}</Text>
    </Pressable>
  );
}

export function Chip({
  label,
  selected,
  onPress,
}: {
  label: string;
  selected: boolean;
  onPress: () => void;
}) {
  return (
    <Pressable
      accessibilityRole="checkbox"
      accessibilityState={{ checked: selected }}
      onPress={onPress}
      style={[styles.chip, selected && styles.chipSelected]}>
      <Text style={[styles.chipText, selected && { color: Colors.white }]}>{label}</Text>
    </Pressable>
  );
}

/** White bar pinned to the bottom of the screen holding the main action. */
export function BottomBar({ children }: { children: ReactNode }) {
  const insets = useSafeAreaInsets();
  return <View style={[styles.bottomBar, { paddingBottom: Math.max(insets.bottom, 16) }]}>{children}</View>;
}

export const styles = StyleSheet.create({
  logo: { fontFamily: Fonts.logo, fontSize: 30, color: Colors.primary, letterSpacing: -0.5 },
  button: {
    backgroundColor: Colors.primary,
    borderRadius: 999,
    paddingVertical: 18,
    alignItems: 'center',
  },
  buttonText: { fontFamily: Fonts.semibold, fontSize: 17, color: Colors.white },
  outline: {
    backgroundColor: Colors.white,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: Colors.border,
    paddingVertical: 17,
    alignItems: 'center',
  },
  outlineText: { fontFamily: Fonts.semibold, fontSize: 16, color: Colors.text },
  ghost: { paddingVertical: 14, alignItems: 'center' },
  ghostText: { fontFamily: Fonts.semibold, fontSize: 15, color: Colors.accent },
  chip: {
    paddingHorizontal: 17,
    paddingVertical: 11,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: Colors.chipBorder,
    backgroundColor: Colors.white,
  },
  chipSelected: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  chipText: { fontFamily: Fonts.medium, fontSize: 15, color: Colors.text },
  bottomBar: {
    backgroundColor: Colors.white,
    paddingHorizontal: 24,
    paddingTop: 14,
    shadowColor: '#000',
    shadowOpacity: 0.04,
    shadowRadius: 12,
    shadowOffset: { width: 0, height: -4 },
    elevation: 4,
  },
});
