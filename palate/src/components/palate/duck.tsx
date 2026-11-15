import { Image } from 'expo-image';
import Svg, { Circle, Path } from 'react-native-svg';

import { Colors } from '@/constants/palette';

// Duck art exported from Figma.
const DUCKS = {
  dot: { source: require('@/assets/images/duck.png'), ratio: 381 / 450 },
  wink: { source: require('@/assets/images/duck-eyes-closed.png'), ratio: 319 / 363 },
};

export function Duck({ width = 220, eye = 'dot' }: { width?: number; eye?: 'dot' | 'wink' }) {
  const { source, ratio } = DUCKS[eye];
  return <Image source={source} style={{ width, height: width * ratio }} contentFit="contain" accessible={false} />;
}

/** Party popper next to the duck on the confirmation screen. */
export function Popper({ size = 34 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 32 32" fill="none">
      <Path d="M4 28 L11 10 L22 21 Z" stroke={Colors.primary} strokeWidth={1.8} strokeLinejoin="round" />
      <Path d="M8 19 L13 24 M10 14 L18 22" stroke={Colors.primary} strokeWidth={1.4} />
      <Path d="M17 9 Q19 5 23 6 M22 13 Q25 11 28 13" stroke={Colors.primary} strokeWidth={1.6} strokeLinecap="round" />
      <Circle cx={15} cy={4} r={1.4} fill={Colors.primary} />
      <Circle cx={27} cy={8} r={1.4} fill={Colors.primary} />
      <Circle cx={26} cy={18} r={1.4} fill={Colors.primary} />
    </Svg>
  );
}

/** Little outline bird used as the input icon on the home screen. */
export function BirdIcon({ size = 22 }: { size?: number }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <Path
        d="M3 9 L9 12 L12 5 L15 12 L21 9 L18 17 Q12 21 6 17 Z"
        stroke={Colors.primary}
        strokeWidth={1.8}
        strokeLinejoin="round"
      />
    </Svg>
  );
}

export function Star({ size = 48, color = Colors.blush }: { size?: number; color?: string }) {
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      <Path d="M12 1 L14.6 8.6 L23 9 L16.3 14 L18.8 22.5 L12 17.4 L5.2 22.5 L7.7 14 L1 9 L9.4 8.6 Z" fill={color} />
    </Svg>
  );
}

export function Blob({ width = 180, color = Colors.sage }: { width?: number; color?: string }) {
  return (
    <Svg width={width} height={width} viewBox="0 0 100 100">
      <Path d="M20 0 H100 V70 Q80 80 70 60 Q55 35 30 35 Q5 30 20 0 Z" fill={color} />
    </Svg>
  );
}
