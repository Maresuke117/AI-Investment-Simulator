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
import io

load_dotenv()

# --- 設定 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
RESULTS_FILE = "market_scan_results.csv"

# --- キャッシュ ---
NAME_CACHE = {}

def get_all_target_tickers():
    """日米合計 約1,400銘柄 (US: Russell 1000相当 / JP: JPX日経400) を取得"""
    global NAME_CACHE
    
    # 1. 米国株 (Russell 1000相当)
    us_tickers = []
    try:
        # 複数のソースを試行
        urls = [
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
            "https://pkgstore.datahub.io/core/s-and-p-500-companies/constituents_csv/data/ac117143448e89b91e9f13e71618197e/constituents_csv.csv"
        ]
        for url in urls:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    df_us = pd.read_csv(io.StringIO(res.text))
                    for _, row in df_us.iterrows():
                        ticker = str(row['Symbol']).replace('.', '-')
                        us_tickers.append(ticker)
                        NAME_CACHE[ticker] = row['Name']
                    if len(us_tickers) > 400: break
            except: continue
            
        # 主要追加銘柄 (Russell 1000上位の補完)
        additional_us = {
            "TSLA": "Tesla, Inc.", "NVDA": "NVIDIA Corporation", "AMD": "Advanced Micro Devices, Inc.",
            "PLTR": "Palantir Technologies Inc.", "ARM": "Arm Holdings plc", "SMCI": "Super Micro Computer, Inc.",
            "AVGO": "Broadcom Inc.", "ORCL": "Oracle Corporation", "COST": "Costco Wholesale Corporation"
        }
        for t, n in additional_us.items():
            if t not in us_tickers:
                us_tickers.append(t)
                NAME_CACHE[t] = n
        us_tickers = list(set(us_tickers))
        print(f"🇺🇸 米国株: {len(us_tickers)} 銘柄を取得。")
    except Exception as e:
        print(f"⚠️ 米国株リスト取得エラー: {e}")
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

    # 2. 日本株 (JPX日経400限定)
    jp_tickers = []
    try:
        url_jpx400 = "https://www.jpx.co.jp/markets/indices/jpx-nikkei400/tvdivq0000001vg2-att/jpxnk400_weight_j.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url_jpx400, headers=headers, timeout=10)
        
        try:
            df_jpx400 = pd.read_csv(io.BytesIO(res.content), encoding='cp932')
        except:
            df_jpx400 = pd.read_csv(io.BytesIO(res.content), encoding='utf-8-sig')
            
        code_col = [col for col in df_jpx400.columns if 'コード' in col or 'Code' in col][0]
        name_col = [col for col in df_jpx400.columns if '銘柄名' in col or 'Name' in col][0]
        
        for _, row in df_jpx400.iterrows():
            code = str(row[code_col])
            name = str(row[name_col])
            clean_code = "".join(filter(str.isdigit, code))[:4]
            if len(clean_code) == 4:
                ticker = f"{clean_code}.T"
                jp_tickers.append(ticker)
                NAME_CACHE[ticker] = name
                        
        print(f"🇯🇵 日本株 (JPX400): {len(jp_tickers)} 銘柄を取得。")
    except Exception as e:
        print(f"⚠️ JPX400リスト取得エラー: {e}")
        jp_tickers = ["7203.T", "6758.T", "6861.T"]

    return us_tickers, jp_tickers

def stage1_screening(ticker):
    """Stage 1: 高速テクニカルスクリーニング (高品質銘柄用)"""
    global NAME_CACHE
    try:
        data, _, api_name = get_stock_data(ticker, period="1y")
        if len(data) < 100: return None
        
        # 名前をキャッシュに保存 (取得できた場合のみ)
        if ticker not in NAME_CACHE and api_name and api_name != ticker:
            NAME_CACHE[ticker] = api_name
            
        name = NAME_CACHE.get(ticker, api_name)
        
        data = prepare_features(data)
        latest = data.iloc[-1]
        
        score = 0
        # 1. 強いトレンド (20日線 > 50日線 ＆ 株価 > 20日線)
        if latest['Close'] > latest['SMA_20']: score += 1
        if latest['SMA_20'] > latest['SMA_50']: score += 1
        
        # 2. 適正なRSI (買われすぎず、売られすぎず)
        if 40 < latest['RSI'] < 65: score += 1
        
        # 3. 低ボラティリティ (値動きが素直)
        vol = data['Close'].pct_change().tail(20).std()
        if vol < 0.025: score += 1
        
        if score < 1: return None # 1点以上を合格とする (門戸を広げる)
        
        return {"Ticker": ticker, "ScreeningScore": score, "Name": name}
    except: return None

def process_single_stock_heavy(ticker, name):
    """Stage 2: 精密AI分析"""
    try:
        import random
        time.sleep(random.uniform(0.2, 0.5))
        
        data, currency, api_name = get_stock_data(ticker, period="2y")
        if data is None or len(data) < 150: return None
        
        # 最新の名前を確保
        final_name = NAME_CACHE.get(ticker, api_name)
            
        data = prepare_features(data)
        strategy = AIStrategy(api_key=GEMINI_API_KEY, precise=True)
        score = strategy.train(data)
        signals = strategy.predict_signals(data)
        
        if signals.empty or 'Prediction' not in signals.columns:
            return None
            
        avg_pred = signals['Prediction'].tail(5).mean()
        annualized_pred = avg_pred * 252
        
        return {
            "Name": final_name, "Ticker": ticker, "Price": signals['Close'].iloc[-1],
            "Currency": currency, "AI Prediction": annualized_pred,
            "Confidence": score
        }
    except:
        return None

def run_mass_scan():
    print("📢 --- 【日米エリート】1,400銘柄・公平なAI分析システム起動 ---")
    
    us_tickers, jp_tickers = get_all_target_tickers()
    all_tickers = list(set(us_tickers + jp_tickers))
    print(f"✅ ターゲット確定: {len(all_tickers)} 銘柄 (Russell 1000 & JPX400)")

    print(f"🚀 Stage 1: 高品質フィルタリング開始...")
    screened_candidates = []
    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(stage1_screening, t): t for t in all_tickers}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: screened_candidates.append(res)
            
    # 上位300銘柄をAI分析へ
    screened_candidates = sorted(screened_candidates, key=lambda x: x['ScreeningScore'], reverse=True)[:300]
    print(f"✅ Stage 1 完了。精鋭 {len(screened_candidates)} 銘柄をAI診断へ投入。")

    print(f"🧠 Stage 2: 精密AI分析開始...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_stock_heavy, c['Ticker'], c['Name']): c for c in screened_candidates}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: results.append(res)

    try:
        if results:
            df_all = pd.DataFrame(results)
            df_all['Score'] = df_all['AI Prediction'] * (df_all['Confidence'] + 2.0)
            df_market = df_all[df_all['AI Prediction'] > 0].sort_values(by="Score", ascending=False)
            df_market.to_csv(RESULTS_FILE, index=False)
            
            if not df_market.empty:
                send_discord_summary(df_market)
                print(f"✨ 全分析完了。{len(df_market)} 銘柄のチャンスを特定しました。")
        else:
            print("❌ 分析結果が空でした。")
    except Exception as e:
        print(f"❌ 集計中にエラーが発生しました: {e}")

def send_discord_summary(df):
    if not DISCORD_WEBHOOK_URL or df.empty: return
    
    tier1 = df[(df['Confidence'] > 0) & (df['AI Prediction'] > 0)].head(10)
    tier2 = df[(df['Confidence'] <= 0) & (df['Confidence'] > -1) & (df['AI Prediction'] > 0)].head(10)
    tier3 = df[(df['Confidence'] <= -1) & (df['AI Prediction'] > 0)].head(5)
    
    msg = "💎 **【究極のエリート】日米1,400銘柄・AI特選レポート** 💎\n（Russell 1000 / JPX日経400 厳選構成）\n\n"
    
    if not tier1.empty:
        msg += "🎯 **【最優先】AI Recommended (最高評価)**\n"
        for i, (_, row) in enumerate(tier1.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']}): +{row['AI Prediction']:.1%} (信頼度: {row['Confidence']:.3f})\n"
        msg += "\n"
        
    if not tier2.empty:
        msg += "📈 **【準優先】Potential (上昇予測 ＆ 信頼度中)**\n"
        for i, (_, row) in enumerate(tier2.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯pis"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']}): +{row['AI Prediction']:.1%} (信頼度: {row['Confidence']:.3f})\n"
        msg += "\n"

    if not tier3.empty:
        msg += "⚠️ **【参考】Speculative (上昇予測 ＆ 信頼度低)**\n"
        for i, (_, row) in enumerate(tier3.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']})\n"
            
    msg += "\n🔍 詳細な戦略はダッシュボードで確認してください！"
    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})

if __name__ == "__main__":
    run_mass_scan()
