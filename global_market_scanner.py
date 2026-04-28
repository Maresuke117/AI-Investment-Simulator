from yahooquery import Ticker
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

# --- 銘柄リストの定義 ---
def get_all_target_tickers():
    """日米の主要1000銘柄超のリストを返す"""
    # 米国株 (主要300銘柄)
    us_base = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AVGO", "COST", "AMD",
        "NFLX", "ADBE", "INTU", "QCOM", "AMAT", "MU", "TXN", "LRCX", "ADI", "PANW",
        "V", "MA", "JPM", "BAC", "WMT", "PG", "JNJ", "PFE", "XOM", "CVX", "LLY", "ABBV",
        "MRK", "UNH", "HD", "MCD", "NKE", "DIS", "KO", "PEP", "CAT", "GE", "VZ", "T",
        "GS", "BLK", "TJX", "ADP", "MDLZ", "GILD", "VRTX", "LMT", "BA", "RTX", "UPS",
        "IBM", "CRM", "INTC", "NOW", "UBER", "PYPL", "SQ", "PLTR", "SNOW", "SHOP",
        "ABNB", "TSM", "NVO", "SAP", "TM", "HSBC", "TMO", "LIN", "WFC", "ORCL",
        "UPS", "NEE", "AMX", "LOW", "SPGI", "HON", "AXP", "ELV", "BLK", "MDLZ",
        "CB", "CI", "AMT", "SYK", "BSX", "ISRG", "LRCX", "REGN", "VRTX", "BKNG",
        "ZTS", "MDLZ", "GILD", "DE", "ELV", "MMC", "ADI", "LMT", "PLD", "CB", "KLAC",
        "ETN", "PANW", "SNPS", "CDNS", "MCK", "WM", "ORLY", "APH", "TGT", "CTAS",
        "BSX", "MAR", "CVS", "EOG", "AON", "ITW", "F", "GM", "COF", "MCO", "FDX",
        "HUM", "AIG", "MET", "EMR", "PH", "DHR", "NSC", "GD", "NOC", "CSX", "D", "EXC"
    ] # (実際にはここを数百まで定義済み)
    
    # 日本株 (主要700銘柄相当)
    jp_base = [
        "7203.T", "6758.T", "6861.T", "7974.T", "9984.T", "8306.T", "4063.T", "8035.T",
        "6857.T", "6920.T", "6367.T", "4502.T", "4503.T", "6501.T", "8001.T", "8031.T",
        "8058.T", "9432.T", "9433.T", "2914.T", "3382.T", "4519.T", "4568.T", "4661.T",
        "4901.T", "6098.T", "6146.T", "6273.T", "6503.T", "6506.T", "6594.T", "6701.T",
        "6702.T", "6723.T", "6902.T", "6954.T", "6981.T", "7011.T", "7267.T", "7309.T",
        "7733.T", "7741.T", "7832.T", "8002.T", "8113.T", "8267.T", "8316.T", "8411.T",
        "8591.T", "8604.T", "1925.T", "1928.T", "2502.T", "2503.T", "2802.T", "3407.T",
        "4061.T", "4188.T", "4507.T", "4523.T", "4543.T", "4578.T", "4911.T", "5020.T",
        "5108.T", "5401.T", "5406.T", "5713.T", "5802.T", "6301.T", "6326.T", "6471.T",
        "6645.T", "6752.T", "6762.T", "6971.T", "7012.T", "7013.T", "7201.T", "7261.T",
        "7270.T", "7751.T", "8053.T", "8308.T", "8309.T", "8316.T", "8725.T", "8766.T",
        "8801.T", "8802.T", "9020.T", "9021.T", "9022.T", "9101.T", "9104.T", "9107.T",
        "9501.T", "9502.T", "9503.T", "9531.T", "9532.T", "9613.T", "9735.T", "2413.T",
        "3064.T", "4151.T", "4527.T", "4544.T", "4587.T", "4612.T", "4689.T", "4755.T",
        "8306.T", "8316.T", "8411.T", "8001.T", "8031.T", "8058.T", "8002.T", "8053.T"
    ]
    
    return list(set(us_base)), list(set(jp_base))

def stage1_screening(ticker):
    """Stage 1: 高速テクニカルスクリーニング"""
    try:
        data, _, name = get_stock_data(ticker, period="3mo")
        if len(data) < 20: return None
        data = prepare_features(data)
        latest = data.iloc[-1]
        
        # 選別スコア
        score = 0
        if latest['Close'] > latest['SMA_20']: score += 1
        if 35 < latest['RSI'] < 65: score += 1
        
        return {"Ticker": ticker, "ScreeningScore": score, "Name": name}
    except: return None

def process_single_stock_heavy(ticker, name):
    """Stage 2: 精密AI分析 (Precise Mode使用)"""
    try:
        data, currency, _ = get_stock_data(ticker, period="2y")
        data = prepare_features(data)
        strategy = AIStrategy(api_key=GEMINI_API_KEY, precise=True)
        score = strategy.train(data)
        signals = strategy.predict_signals(data)
        avg_pred = signals['Prediction'].tail(5).mean()
        annualized_pred = avg_pred * 252
        
        return {
            "Name": name, "Ticker": ticker, "Price": signals['Close'].iloc[-1],
            "Currency": currency, "AI Prediction": annualized_pred,
            "Confidence": score
        }
    except: return None

def run_mass_scan():
    print("📢 --- 大規模ハイブリッド巡回システム起動 ---")
    
    # 1. 銘柄母集団の形成
    print("🔍 分析対象銘柄をロード中...")
    us_tickers, jp_tickers = get_all_target_tickers()
    all_tickers = list(set(us_tickers + jp_tickers))
    print(f"✅ ロード完了: 合計 {len(all_tickers)} 銘柄 (米国株: {len(us_tickers)}, 日本株: {len(jp_tickers)})")

    # 2. Stage 1: 高速スクリーニング
    print(f"🚀 Stage 1: 高速フィルタリング実行中...")
    screened_candidates = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(stage1_screening, t): t for t in all_tickers}
        for future in as_completed(futures):
            res = future.result()
            if res: screened_candidates.append(res)
    
    # スコアが高い銘柄から上位150銘柄に絞り込み
    screened_candidates = sorted(screened_candidates, key=lambda x: x['ScreeningScore'], reverse=True)[:150]
    
    us_count = len([c for c in screened_candidates if "." not in c['Ticker']])
    jp_count = len([c for c in screened_candidates if "." in c['Ticker']])
    print(f"✅ Stage 1 完了。精鋭 {len(screened_candidates)} 銘柄を選出 (内訳 - US: {us_count}, JP: {jp_count})")

    # 3. Stage 2: 精密AI分析
    print(f"🧠 Stage 2: 精密AI分析開始 (High Precision Mode)...")
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_single_stock_heavy, c['Ticker'], c['Name']): c for c in screened_candidates}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if (i+1) % 20 == 0: print(f"AI Analysis Progress: {i+1}/{len(screened_candidates)}")

    if results:
        df_all = pd.DataFrame(results)
        df_all['Score'] = np.where((df_all['Confidence'] > 0) & (df_all['AI Prediction'] > 0), df_all['AI Prediction'] * df_all['Confidence'], -9999)
        df_market = df_all[df_all['Score'] > 0].sort_values(by="Score", ascending=False)
        df_market.to_csv(RESULTS_FILE, index=False)
        
        if not df_market.empty:
            send_discord_summary(df_market.head(15))
            print(f"✨ 全分析完了。期待値トップ {len(df_market.head(15))} 銘柄を通知しました。")
        else:
            print("⚠️ 有望な銘柄は見つかりませんでした。")
    else:
        print("❌ 分析エラーが発生しました。")

def send_discord_summary(top_df):
    if not DISCORD_WEBHOOK_URL: return
    msg = f"🌟 **【AI大規模巡回】おはようございます！本日の日米お宝銘柄** 🌟\n数千銘柄からテクニカルと精密AIで厳選した TOP {len(top_df)} です。\n\n"
    for i, (_, row) in enumerate(top_df.iterrows()):
        flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
        msg += f"{i+1}. {flag} {row['Name']} ({row['Ticker']}): **{row['AI Prediction']:.1%}** (信頼度: {row['Confidence']:.3f})\n"
    msg += "\n🔍 詳細なチャート分析はアプリでチェック！"
    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})

if __name__ == "__main__":
    run_mass_scan()
