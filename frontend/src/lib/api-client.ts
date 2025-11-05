/**
 * API Client for Trading Platform
 * トレーディングプラットフォームのAPIクライアント
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl: number;
  pnl_percent: number;
  entry_time: string;
  exit_time: string;
  duration: number;
}

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
}

interface BacktestResult {
  id: string;
  strategy_name: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  created_at: string;
}

interface HealthStatus {
  status: string;
  version: string;
  timestamp: string;
  uptime: number;
  database: string;
  cache: string;
}

export class ApiClient {
  private client: AxiosInstance;
  private token: string | null = null;

  constructor(baseURL: string = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.client.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      return config;
    });

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        if (error.response?.status === 401) {
          // Handle unauthorized
          this.clearAuth();
        }
        return Promise.reject(error);
      }
    );
  }

  setToken(token: string) {
    this.token = token;
  }

  clearAuth() {
    this.token = null;
  }

  // Health & Status
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  // Trades
  async getTrades(limit: number = 100): Promise<Trade[]> {
    const response = await this.client.get<ApiResponse<Trade[]>>('/api/trades', {
      params: { limit },
    });
    return response.data.data || [];
  }

  async getTrade(tradeId: string): Promise<Trade | null> {
    try {
      const response = await this.client.get<ApiResponse<Trade>>(`/api/trades/${tradeId}`);
      return response.data.data || null;
    } catch (error) {
      console.error('Failed to get trade:', error);
      return null;
    }
  }

  async getTradeStats(): Promise<{
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_pnl: number;
    avg_win: number;
    avg_loss: number;
  }> {
    const response = await this.client.get('/api/trades/stats');
    return response.data.data;
  }

  // Positions
  async getPositions(): Promise<Position[]> {
    const response = await this.client.get<ApiResponse<Position[]>>('/api/positions');
    return response.data.data || [];
  }

  async openPosition(symbol: string, side: 'long' | 'short', quantity: number, entryPrice: number): Promise<Position> {
    const response = await this.client.post<ApiResponse<Position>>('/api/positions', {
      symbol,
      side,
      quantity,
      entry_price: entryPrice,
    });
    return response.data.data!;
  }

  async closePosition(positionId: string, exitPrice: number): Promise<Trade> {
    const response = await this.client.post<ApiResponse<Trade>>(`/api/positions/${positionId}/close`, {
      exit_price: exitPrice,
    });
    return response.data.data!;
  }

  // Backtesting
  async getBacktestResults(limit: number = 50): Promise<BacktestResult[]> {
    const response = await this.client.get<ApiResponse<BacktestResult[]>>('/api/backtests', {
      params: { limit },
    });
    return response.data.data || [];
  }

  async runBacktest(strategyName: string, parameters: Record<string, any>): Promise<BacktestResult> {
    const response = await this.client.post<ApiResponse<BacktestResult>>('/api/backtests/run', {
      strategy_name: strategyName,
      parameters,
    });
    return response.data.data!;
  }

  // Portfolio
  async getPortfolio(): Promise<{
    total_capital: number;
    current_value: number;
    total_pnl: number;
    total_pnl_percent: number;
    cash: number;
  }> {
    const response = await this.client.get('/api/portfolio');
    return response.data.data;
  }

  // Authentication
  async login(email: string, password: string): Promise<{ access_token: string; refresh_token: string }> {
    const response = await this.client.post('/api/auth/login', {
      email,
      password,
    });
    const { access_token, refresh_token } = response.data.data;
    this.setToken(access_token);
    return { access_token, refresh_token };
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/api/auth/logout');
    } finally {
      this.clearAuth();
    }
  }

  async refreshToken(refreshToken: string): Promise<string> {
    const response = await this.client.post('/api/auth/refresh', {
      refresh_token: refreshToken,
    });
    const { access_token } = response.data.data;
    this.setToken(access_token);
    return access_token;
  }

  // Market Data
  async getMarketData(symbol: string, interval: string = '1h'): Promise<any[]> {
    const response = await this.client.get(`/api/market/${symbol}/ohlcv`, {
      params: { interval },
    });
    return response.data.data || [];
  }

  async getPrice(symbol: string): Promise<number> {
    const response = await this.client.get(`/api/market/${symbol}/price`);
    return response.data.data.price;
  }

  // Strategies
  async getStrategies(): Promise<any[]> {
    const response = await this.client.get('/api/strategies');
    return response.data.data || [];
  }

  async getStrategy(strategyId: string): Promise<any> {
    const response = await this.client.get(`/api/strategies/${strategyId}`);
    return response.data.data;
  }

  async updateStrategy(strategyId: string, config: Record<string, any>): Promise<any> {
    const response = await this.client.put(`/api/strategies/${strategyId}`, config);
    return response.data.data;
  }
}

// Export singleton instance
export const apiClient = new ApiClient();
