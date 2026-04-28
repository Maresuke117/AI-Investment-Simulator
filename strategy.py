import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import xgboost as xgb
import google.generativeai as genai
from openai import AzureOpenAI
import os
from stable_baselines3 import PPO
from rl_env import TradingEnv

def prepare_features(df):
    """
    テクニカル指標を特徴量として追加する。
    """
    df = df.copy()
    
    # 移動平均
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ボリンジャーバンド
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['Upper_BB'] = df['SMA_20'] + (df['Std_20'] * 2)
    df['Lower_BB'] = df['SMA_20'] - (df['Std_20'] * 2)
    
    # 変化率 (Returns)
    df['Target'] = df['Close'].shift(-1) / df['Close'] - 1 # 翌日の収益率
    
    # 欠損値の削除
    df = df.dropna()
    
    return df

class AIStrategy:
    def __init__(self, api_key=None):
        # XGBoostのハイパーパラメータ設定 (より精度の高い設定)
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'Upper_BB', 'Lower_BB']
        
        # LLM初期化 (Azure OpenAIを優先)
        self.llm = None
        self.llm_type = None
        
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

        if azure_key and azure_endpoint and azure_deployment:
            try:
                self.llm = AzureOpenAI(
                    api_key=azure_key,
                    api_version=azure_version,
                    azure_endpoint=azure_endpoint
                )
                self.llm_type = "azure"
                self.deployment_name = azure_deployment
                print(f"Using Azure OpenAI: {azure_deployment}")
            except Exception as e:
                print(f"Azure OpenAI Setup Error: {e}")
        
        if not self.llm and api_key:
            try:
                genai.configure(api_key=api_key)
                models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                selected_model = 'models/gemini-1.5-flash'
                if not any('gemini-1.5-flash' in m for m in models):
                    selected_model = models[0] if models else 'gemini-pro'
                
                print(f"Using Gemini Model: {selected_model}")
                self.llm = genai.GenerativeModel(selected_model)
                self.llm_type = "gemini"
            except Exception as e:
                print(f"Gemini Setup Error: {e}")

    def train(self, df):
        """
        XGBoostモデルを学習させる。
        """
        df_feat = prepare_features(df)
        X = df_feat[self.features]
        y = df_feat['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        score = self.model.score(X_test, y_test)
        print(f"XGBoost Model R^2 Score: {score:.4f}")
        return score

    def predict_signals(self, df):
        """
        最新のデータに基づいてシグナルを生成する。
        """
        df_feat = prepare_features(df)
        X = df_feat[self.features]
        
        # 予測
        predictions = self.model.predict(X)
        df_feat['Prediction'] = predictions
        
        # シグナル: 予測収益率 > 0.2% なら 買(1)
        df_feat['Signal'] = np.where(df_feat['Prediction'] > 0.002, 1, 0)
        
        return df_feat

    def get_sentiment(self, ticker, news_text):
        """
        Gemini Proを使用してニュースのセンチメントを解析する。
        """
        if not self.llm:
            return 0, "APIキーが設定されていません。"
            
        prompt = f"""
        あなたはプロの証券アナリストです。
        以下の {ticker} に関する最新ニュースを読み、今後1週間〜1ヶ月の株価に対する影響を -1.0（非常に弱気）から 1.0（非常に強気）の数値でスコアリングしてください。
        
        【スコアリング基準】
        - +0.7以上: 非常に強いポジティブ材料（好決算、大型受注、画期的な新製品など）
        - +0.2〜+0.7: 良好な材料（順調な進捗、市場環境の改善など）
        - -0.2〜+0.2: 影響が軽微、または材料が混合している
        - -0.7〜-0.2: 懸念材料（競合の台頭、コスト増、成長鈍化など）
        - -1.0〜-0.7: 深刻な悪材料（不祥事、大幅な下方修正、致命的な製品欠陥など）
        
        【出力フォーマット（JSON形式）】
        {{
            "score": 0.5,
            "reason": "なぜそのスコアにしたのか、具体的なニュース内容に触れながら日本語で解説してください。"
        }}
        
        ニュース内容:
        {news_text}
        """
        try:
            if self.llm_type == "azure":
                response = self.llm.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            elif self.llm_type == "gemini":
                response = self.llm.generate_content(prompt)
                return response.text
            return "LLM is not configured."
        except Exception as e:
            return f"Error: {e}"

    def get_advice(self, current_price, buy_price, prediction):
        """
        保有銘柄に対するアドバイスを生成する。
        """
        profit_pct = (current_price / buy_price - 1)
        
        if profit_pct > 0.30: # 30%以上の利益
            if prediction < 0:
                return "利確推奨 (Target達成 & 下落予測)", "warning"
            else:
                return "ホールド推奨 (Target達成だが上昇余地あり)", "success"
        
        if profit_pct < -0.10: # 10%以上の損失（損切り）
            return "損切り検討 (下落リスク)", "error"
            
        if prediction > 0.002: # 強い上昇予測
            return "ホールド (上昇トレンド継続予測)", "success"
        elif prediction < -0.002:
            return "一部利確または警戒 (短期下落予測)", "warning"
        
        return "継続ホールド (横ばい予測)", "info"

    def train_rl(self, df, total_timesteps=500):
        """
        銘柄のボラティリティと指標に基づき、売買戦略（利確・損切）を高速に最適化する。
        """
        df_clean = df.dropna()
        if len(df_clean) < 50: 
            self.best_params = {"profit_target": 0.15, "stop_loss": -0.1, "rsi_sell": 0.8}
            self.rl_ready = True
            return
        
        # 銘柄の標準的な変動（ボラティリティ）を計算
        returns = df_clean['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) # 年率ボラ
        
        # ボラティリティに応じて探索範囲を調整
        p_targets = [0.05, 0.1, 0.2, 0.3] if volatility < 0.2 else [0.1, 0.2, 0.3, 0.5, 0.8]
        s_losses = [-0.03, -0.07, -0.1] if volatility < 0.2 else [-0.1, -0.15, -0.25]
        rsi_targets = [0.7, 0.75, 0.8, 0.85]
        
        best_reward = -float('inf')
        self.best_params = {"profit_target": 0.2, "stop_loss": -0.1, "rsi_sell": 0.8}
        
        # 過去1年間のデータで「最も稼げたルール」を探す
        test_data = df_clean.tail(250)
        for pt in p_targets:
            for sl in s_losses:
                for rt in rsi_targets:
                    total_ret = 0
                    # サンプリングシミュレーション
                    for start_idx in range(0, len(test_data)-20, 10):
                        entry_p = test_data['Close'].iloc[start_idx]
                        for j in range(start_idx+1, min(start_idx+20, len(test_data))):
                            cur_p = test_data['Close'].iloc[j]
                            cur_rsi = test_data['RSI'].iloc[j] / 100.0
                            ret = (cur_p / entry_p) - 1
                            if ret >= pt or ret <= sl or cur_rsi >= rt:
                                total_ret += ret
                                break
                    
                    if total_ret > best_reward:
                        best_reward = total_ret
                        self.best_params = {"profit_target": pt, "stop_loss": sl, "rsi_sell": rt}
        
        self.rl_ready = True

    def get_rl_advice(self, df, buy_price):
        """
        最適化された戦略と現在のRSI等の指標に基づいた判断を返す。
        """
        if not hasattr(self, 'rl_ready'):
            return "未学習です", "info"
            
        current_price = df['Close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1] / 100.0
        profit_pct = (current_price / buy_price) - 1
        
        pt = self.best_params["profit_target"]
        sl = self.best_params["stop_loss"]
        rt = self.best_params["rsi_sell"]
        
        # 判断ロジック
        if profit_pct >= pt:
            return f"AI売却推奨 (目標利益 {pt:.0%} 達成)", "error"
        if current_rsi >= rt:
            return f"AI売却推奨 (RSI過熱感 {current_rsi:.0%} 到達)", "error"
        if profit_pct <= sl:
            return f"AI損切推奨 (損切ライン {sl:.0%} 到達)", "error"
            
        # 期待値によるホールド
        return f"AIホールド推奨 (目標 {pt:.0%} / RSI {rt:.0%} まで待機)", "success"

    def get_investment_advice(self, ticker, current_price, prediction, sentiment_score, budget, currency_symbol):
        """
        予算と予測、センチメントに基づいた具体的な投資配分のアドバイスを生成する。
        """
        if not self.llm:
            return "APIキーが設定されていないため、詳細なアドバイスを生成できません。"
            
        prompt = f"""
        あなたはプロの資産運用アドバイザーです。
        以下のデータに基づき、この銘柄に対して「いくら投資すべきか」とその「戦略的理由」を提案してください。
        
        【データ】
        - 銘柄: {ticker}
        - 現在価格: {currency_symbol}{current_price:,.2f}
        - AIによる年間期待収益率予測: {prediction:.1%}
        - ニュースのセンチメントスコア (-1.0〜1.0): {sentiment_score:.2f}
        - ユーザーの総投資予算: {currency_symbol}{budget:,.0f}
        
        【出力ルール】
        1. 推奨投資金額: 予算の何%（および具体的な金額）をこの銘柄に割り当てるべきか。
        2. 戦略的理由: なぜその金額なのか、リスクとリターンのバランスを考慮して解説。
        3. 出口戦略: どのような状態になったら売却すべきか（利確・損切りの目安）。
        
        日本語で、プロフェッショナルかつ分かりやすく回答してください。
        """
        try:
            if self.llm_type == "azure":
                response = self.llm.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            elif self.llm_type == "gemini":
                response = self.llm.generate_content(prompt)
                return response.text
            return "LLM is not configured."
        except Exception as e:
            return f"Advice generation error: {e}"

if __name__ == "__main__":
    from data_loader import get_stock_data
    
    data = get_stock_data("AAPL")
    strategy = AIStrategy()
    strategy.train(data)
    signals = strategy.predict_signals(data)
    print(signals[['Close', 'Prediction', 'Signal']].tail())
