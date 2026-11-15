// Colors and fonts pulled from the Figma screens in /designs.

export const Colors = {
  cream: '#F9F5F1',
  white: '#FFFFFF',
  primary: '#D4775F', // terracotta buttons, selected chips, logo
  primaryDark: '#B5604A',
  accent: '#B07457', // section labels, "See more"
  text: '#2E2A26',
  muted: '#8A8178',
  border: '#E5DCD4',
  inputBg: '#F8F3EF',
  chipBorder: '#E3DAD2',
  addPersonBg: '#F1E6DF',
  progressDone: '#B07457',
  progressCurrent: '#DDB89F',
  progressTodo: '#E6DDD5',
  error: '#C0412B',

  // illustration
  sage: '#DDE0CF',
  olive: '#6F7B40',
  mustard: '#C9A040',
  beakTop: '#DDAA3E',
  beakBottom: '#CD5A3A',
  ink: '#1E1B17',
  blush: '#E3B7A3',
  sand: '#EFDFCF',
  ivory: '#F6EEDD',
} as const;

export const Fonts = {
  regular: 'Inter_400Regular',
  medium: 'Inter_500Medium',
  semibold: 'Inter_600SemiBold',
  bold: 'Inter_700Bold',
  logo: 'Fraunces_700Bold',
} as const;
