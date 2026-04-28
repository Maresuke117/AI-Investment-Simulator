import yfinance as yf
import pandas as pd
import numpy as np
from strategy import AIStrategy, prepare_features
from data_loader import get_stock_data
import os
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()

# --- 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
RESULTS_FILE = "market_scan_results.csv"
PORTFOLIO_FILE = "portfolio.csv"
PORTFOLIO_RESULTS_FILE = "portfolio_scan_results.csv"

# --- 銘柄リストの拡張 (日米主要300銘柄超) ---
TICKERS_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AVGO", "COST", "AMD",
    "NFLX", "ADBE", "INTU", "QCOM", "AMAT", "MU", "TXN", "LRCX", "ADI", "PANW",
    "SNPS", "CDNS", "ASML", "CSCO", "ORCL", "IBM", "CRM", "INTC", "NOW", "UBER",
    "V", "MA", "JPM", "BAC", "WMT", "PG", "JNJ", "PFE", "XOM", "CVX", "LLY", "ABBV",
    "MRK", "UNH", "HD", "MCD", "NKE", "DIS", "KO", "PEP", "CAT", "GE", "VZ", "T",
    "AMD", "INTC", "MU", "AMAT", "LRCX", "KLAC", "TSM", "ARM", "PLTR", "SNOW", "SQ", "PYPL"
] # ここにさらにS&P500銘柄を順次追加可能

TICKERS_JP = [
    "7203.T", "6758.T", "6861.T", "7974.T", "9984.T", "8306.T", "4063.T", "8035.T",
    "6857.T", "6920.T", "6367.T", "4502.T", "4503.T", "6501.T", "8001.T", "8031.T",
    "8058.T", "9432.T", "9433.T", "2914.T", "3382.T", "4519.T", "4568.T", "4661.T",
    "4901.T", "6098.T", "6146.T", "6273.T", "6503.T", "6506.T", "6594.T", "6701.T",
    "6702.T", "6723.T", "6902.T", "6954.T", "6981.T", "7011.T", "7267.T", "7309.T",
    "7733.T", "7741.T", "7832.T", "8002.T", "8113.T", "8267.T", "8316.T", "8411.T",
    "8591.T", "8604.T"
]

def process_single_stock(ticker):
    """1銘柄のAI解析を実行"""
    try:
        data, currency = get_stock_data(ticker, period="2y")
        if len(data) < 100: return None
        data = prepare_features(data)
        strategy = AIStrategy(api_key=GEMINI_API_KEY)
        # 1. XGBoost 学習
        score = strategy.train(data)
        signals = strategy.predict_signals(data)
        avg_pred = signals['Prediction'].tail(5).mean()
        annualized_pred = avg_pred * 252
        try:
            name = yf.Ticker(ticker).info.get('shortName', ticker)
        except:
            name = ticker
        return {
            "Name": name, "Ticker": ticker, "Price": signals['Close'].iloc[-1],
            "Currency": currency, "AI Prediction": annualized_pred,
            "Confidence": score
        }
    except: return None

def run_mass_scan():
    # ポートフォリオ銘柄の読み込み
    p_tickers = []
    if os.path.exists(PORTFOLIO_FILE):
        try:
            p_df = pd.read_csv(PORTFOLIO_FILE)
            p_tickers = p_df['Ticker'].tolist()
        except: pass

    all_target_tickers = list(set(TICKERS_US + TICKERS_JP + p_tickers))
    print(f"🚀 Starting mass scan for {len(all_target_tickers)} tickers...")
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {executor.submit(process_single_stock, t): t for t in all_target_tickers}
        for i, future in enumerate(as_completed(future_to_ticker)):
            res = future.result()
            if res: results.append(res)
            if (i+1) % 20 == 0: print(f"Progress: {i+1}/{len(all_target_tickers)}")

    if results:
        df_all = pd.DataFrame(results)
        
        # 1. 市場全体のランキング保存 (ConfidenceとPredictionでソート)
        df_market = df_all[df_all['Ticker'].isin(TICKERS_US + TICKERS_JP)].sort_values(by=["Confidence", "AI Prediction"], ascending=[False, False])
        df_market.to_csv(RESULTS_FILE, index=False)
        
        # 2. ポートフォリオの診断結果保存
        df_portfolio = df_all[df_all['Ticker'].isin(p_tickers)]
        df_portfolio.to_csv(PORTFOLIO_RESULTS_FILE, index=False)
        
        # Discord通知
        send_discord_summary(df_market.head(10), df_portfolio)
    else:
        print("❌ No results found.")

def send_discord_summary(top_df, p_df):
    if not DISCORD_WEBHOOK_URL:
        print("⚠️ DISCORD_WEBHOOK_URL is not set. Skipping notification.")
        return
    
    print(f"📡 Sending summary to Discord... (Top {len(top_df)} stocks, {len(p_df)} portfolio stocks)")
    
    msg = "⭐ **【AI自動巡回】おはようございます！本日の投資レポートです** ⭐\n\n"
    
    msg += "📈 **市場全体の有望銘柄 TOP 10**\n"
    for i, (_, row) in enumerate(top_df.iterrows()):
        msg += f"{i+1}. {row['Name']} ({row['Ticker']}): **{row['AI Prediction']:.1%}** (信頼度: {row['Confidence']:.3f})\n"
    
    if not p_df.empty:
        msg += "\n💼 **あなたのポートフォリオ診断**\n"
        for _, row in p_df.iterrows():
            status = "🔥 絶好調" if row['AI Prediction'] > 0.2 else "✅ 維持" if row['AI Prediction'] > 0 else "⚠️ 注意"
            msg += f"- {row['Ticker']}: {status} (予測: {row['AI Prediction']:+.1%})\n"
    
    msg += "\n詳細な分析はアプリでチェックしてください！\n"
    msg += "🔗 https://ai-investment-simulator.streamlit.app/"
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})
        if response.status_code == 204 or response.status_code == 200:
            print("✅ Discord notification sent successfully!")
        else:
            print(f"❌ Failed to send Discord notification. Status Code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"❌ Error sending Discord notification: {e}")

if __name__ == "__main__":
    run_mass_scan()
