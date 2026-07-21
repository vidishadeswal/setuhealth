import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  login as apiLogin,
  setAuthToken,
  readStoredAuth,
  AUTH_STORAGE_KEY,
  type StoredAuth,
} from '../api/client';
import type { Role } from '../api/types';

interface AuthContextValue {
  token: string | null;
  role: Role | null;
  email: string | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [auth, setAuth] = useState<StoredAuth | null>(() => readStoredAuth());

  useEffect(() => {
    setAuthToken(auth?.token ?? null);
  }, [auth]);

  const value = useMemo<AuthContextValue>(
    () => ({
      token: auth?.token ?? null,
      role: auth?.role ?? null,
      email: auth?.email ?? null,
      isAuthenticated: auth !== null,
      login: async (email: string, password: string) => {
        const res = await apiLogin(email, password);
        const next: StoredAuth = { token: res.access_token, role: res.role, email };
        localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
        setAuth(next);
      },
      logout: () => {
        localStorage.removeItem(AUTH_STORAGE_KEY);
        setAuth(null);
      },
    }),
    [auth],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
