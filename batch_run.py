import os
import pandas as pd
from yahooquery import Ticker
from data_loader import get_stock_data, prepare_features
from strategy import AIStrategy
import requests
from dotenv import load_dotenv

# .envの読み込み
load_dotenv()

# 設定
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
PORTFOLIO_FILE = "portfolio.csv"

def send_notification(message):
    """Discordに通知を送る"""
    if DISCORD_WEBHOOK_URL:
        payload = {"content": message}
        requests.post(DISCORD_WEBHOOK_URL, json=payload)
    else:
        print("Discord Webhook URL is not set. Outputting to console:")
        print(message)

def run_daily_batch():
    if not os.path.exists(PORTFOLIO_FILE):
        print("Portfolio file not found.")
        return

    portfolio_df = pd.read_csv(PORTFOLIO_FILE)
    if portfolio_df.empty:
        print("Portfolio is empty.")
        return

    strategy = AIStrategy(api_key=os.getenv("GEMINI_API_KEY"))
    report = "📢 **AI Daily Stock Analysis Report** 📢\n"
    report += "------------------------------------------\n"

    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        buy_price = row['Buy Price']
        
        try:
            # 1. データ取得と特徴量生成
            data, currency = get_stock_data(ticker, period="2y")
            data = prepare_features(data)
            
            # 2. XGBoost 予測
            strategy.train(data)
            signals = strategy.predict_signals(data)
            current_price = signals['Close'].iloc[-1]
            prediction = signals['Prediction'].iloc[-1]
            profit_pct = (current_price / buy_price - 1) * 100
            
            # 3. AIアドバイス (XGBoost)
            advice, _ = strategy.get_advice(current_price, buy_price, prediction)
            
            # 4. 強化学習 (RL)
            strategy.train_rl(data, total_timesteps=2000)
            rl_advice, _ = strategy.get_rl_advice(data, buy_price)
            
            # 5. AI センチメント
            t = Ticker(ticker)
            news = t.news
            s_score = "N/A"
            if news:
                news_text = "\n".join([n.get('title', '') for n in news[:5]])
                try:
                    import json
                    raw_s = strategy.get_sentiment(ticker, news_text)
                    if "```json" in raw_s:
                        raw_s = raw_s.split("```json")[1].split("```")[0].strip()
                    res = json.loads(raw_s)
                    s_score = f"{res.get('score', 0):+.2f}"
                except: pass

            # レポートに追加
            report += f"**[{ticker}]**\n"
            report += f"・現在価格: {current_price:,.2f} ({profit_pct:+.2f}%)\n"
            report += f"・AI Advice: {advice}\n"
            report += f"・RL Strategy: {rl_advice}\n"
            report += f"・Sentiment: {s_score}\n"
            report += "------------------------------------------\n"
            
        except Exception as e:
            report += f"⚠️ {ticker} の分析に失敗しました: {e}\n"

    send_notification(report)

if __name__ == "__main__":
    run_daily_batch()
