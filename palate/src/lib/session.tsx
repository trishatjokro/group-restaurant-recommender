import { createContext, use, useState, type ReactNode } from 'react';

// Mirrors the participant dictionaries in the flowchart (Flowchart 2/3).
export type Participant = {
  name: string;
  cuisines: string[];
  specifics: string[];
  prices: number[];
  diets: string[];
  avoids: string[];
  done: boolean;
};

export type Answers = Pick<Participant, 'cuisines' | 'specifics' | 'prices' | 'diets' | 'avoids'>;

export type Session = {
  name: string;
  participants: Participant[];
  current: number; // index of the person whose turn it is
  chosenId: string | null;
};

type SessionContextValue = {
  session: Session | null;
  start: (name: string, names: string[]) => void;
  submitAnswers: (answers: Answers) => void;
  choose: (restaurantId: string) => void;
  reset: () => void;
};

const SessionContext = createContext<SessionContextValue | null>(null);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);

  const value: SessionContextValue = {
    session,
    start: (name, names) =>
      setSession({
        name: name.trim() || 'Dinner',
        current: 0,
        chosenId: null,
        participants: names.map((n) => ({
          name: n.trim(),
          cuisines: [],
          specifics: [],
          prices: [],
          diets: [],
          avoids: [],
          done: false,
        })),
      }),
    submitAnswers: (answers) =>
      setSession((s) => {
        if (!s) return s;
        const participants = s.participants.map((p, i) => (i === s.current ? { ...p, ...answers, done: true } : p));
        return { ...s, participants, current: Math.min(s.current + 1, participants.length) };
      }),
    choose: (restaurantId) => setSession((s) => (s ? { ...s, chosenId: restaurantId } : s)),
    reset: () => setSession(null),
  };

  return <SessionContext value={value}>{children}</SessionContext>;
}

export function useSession() {
  const ctx = use(SessionContext);
  if (!ctx) throw new Error('useSession must be used inside SessionProvider');
  return ctx;
}
