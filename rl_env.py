import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    AIが売買タイミングを学習するためのカスタム環境。
    """
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        # 最低限のデータ量チェック
        if len(df) < 30:
            # ダミーデータでエラーを回避
            df = pd.DataFrame(index=range(30), data={'Close': 1.0, 'RSI': 0.5, 'SMA_20': 1.0, 'Lower_BB': 0.0, 'Upper_BB': 2.0})
            
        self.df = df.reset_index()
        self.current_step = 0
        
        # 状態空間: [保有利益率, 経過日数, RSI, SMA乖離率, ボリンジャーバンド位置]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        # 行動空間: 0 (Hold), 1 (Sell)
        self.action_space = spaces.Discrete(2)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ランダムな位置からシミュレーションを開始（学習の多様性のため）
        self.current_step = np.random.randint(0, len(self.df) - 20)
        self.start_price = self.df.loc[self.current_step, 'Close']
        self.day_count = 0
        return self._get_observation(), {}

    def _get_observation(self):
        current_price = self.df.loc[self.current_step, 'Close']
        profit_pct = (current_price / self.start_price) - 1
        
        # テクニカル指標の取得（前もって計算されている想定）
        rsi = self.df.loc[self.current_step, 'RSI'] / 100.0
        sma_ratio = current_price / self.df.loc[self.current_step, 'SMA_20']
        bb_pos = (current_price - self.df.loc[self.current_step, 'Lower_BB']) / \
                 (self.df.loc[self.current_step, 'Upper_BB'] - self.df.loc[self.current_step, 'Lower_BB'] + 1e-8)
        
        return np.array([profit_pct, self.day_count / 30.0, rsi, sma_ratio, bb_pos], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        self.day_count += 1
        
        current_price = self.df.loc[self.current_step, 'Close']
        profit_pct = (current_price / self.start_price) - 1
        
        done = False
        truncated = False
        reward = 0
        
        if action == 1: # Sell
            done = True
            reward = profit_pct * 100 # 利益率を報酬にする
            if reward > 30: reward *= 1.5 # 30%超えにはボーナス
        elif self.current_step >= len(self.df) - 1:
            done = True
            reward = profit_pct * 100
        else:
            # 継続保有（Hold）
            # 長すぎるとペナルティ（資金効率のため）
            reward = -0.01 
            
        if self.day_count > 60: # 2ヶ月持っても売らない場合は強制終了
            done = True
            truncated = True
        
        return self._get_observation(), reward, done, truncated, {}
