/**
 * Global State Management (Zustand)
 * グローバル状態管理
 */

import { create } from 'zustand';
import { apiClient } from './api-client';

interface AuthState {
  isAuthenticated: boolean;
  user: any | null;
  accessToken: string | null;
  refreshToken: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  setTokens: (access: string, refresh: string) => void;
}

interface DashboardState {
  trades: any[];
  positions: any[];
  portfolio: any;
  selectedTrade: string | null;
  refreshTrades: () => Promise<void>;
  refreshPositions: () => Promise<void>;
  refreshPortfolio: () => Promise<void>;
  selectTrade: (tradeId: string | null) => void;
}

interface RealTimeState {
  priceUpdates: Record<string, number>;
  updatePrice: (symbol: string, price: number) => void;
  clearPrices: () => void;
}

// Auth Store
export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: false,
  user: null,
  accessToken: null,
  refreshToken: null,
  
  login: async (email: string, password: string) => {
    try {
      const { access_token, refresh_token } = await apiClient.login(email, password);
      set({
        isAuthenticated: true,
        accessToken: access_token,
        refreshToken: refresh_token,
      });
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  },
  
  logout: async () => {
    try {
      await apiClient.logout();
    } finally {
      set({
        isAuthenticated: false,
        user: null,
        accessToken: null,
        refreshToken: null,
      });
    }
  },
  
  setTokens: (access: string, refresh: string) => {
    apiClient.setToken(access);
    set({
      isAuthenticated: true,
      accessToken: access,
      refreshToken: refresh,
    });
  },
}));

// Dashboard Store
export const useDashboardStore = create<DashboardState>((set) => ({
  trades: [],
  positions: [],
  portfolio: null,
  selectedTrade: null,
  
  refreshTrades: async () => {
    try {
      const trades = await apiClient.getTrades(100);
      set({ trades });
    } catch (error) {
      console.error('Failed to refresh trades:', error);
    }
  },
  
  refreshPositions: async () => {
    try {
      const positions = await apiClient.getPositions();
      set({ positions });
    } catch (error) {
      console.error('Failed to refresh positions:', error);
    }
  },
  
  refreshPortfolio: async () => {
    try {
      const portfolio = await apiClient.getPortfolio();
      set({ portfolio });
    } catch (error) {
      console.error('Failed to refresh portfolio:', error);
    }
  },
  
  selectTrade: (tradeId: string | null) => {
    set({ selectedTrade: tradeId });
  },
}));

// Real-time Store
export const useRealTimeStore = create<RealTimeState>((set) => ({
  priceUpdates: {},
  
  updatePrice: (symbol: string, price: number) => {
    set((state) => ({
      priceUpdates: {
        ...state.priceUpdates,
        [symbol]: price,
      },
    }));
  },
  
  clearPrices: () => {
    set({ priceUpdates: {} });
  },
}));
