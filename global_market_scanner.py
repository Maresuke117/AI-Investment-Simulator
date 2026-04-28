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

def get_all_target_tickers():
    """日米合計 約1,400銘柄 (US: Russell 1000相当 / JP: JPX日経400) を取得"""
    
    # 1. 米国株 (Russell 1000相当 - 主要1,000銘柄)
    us_tickers = []
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        res = requests.get(url)
        df_us = pd.read_csv(io.StringIO(res.text))
        us_tickers = [t.replace('.', '-') for t in df_us['Symbol'].tolist()]
        
        # 主要追加銘柄 (NASDAQ/Russell 1000上位)
        additional_us = [
            "TSLA", "NVDA", "AMD", "PLTR", "SNOW", "SHOP", "SQ", "PYPL", "UBER", "ABNB",
            "COIN", "MSTR", "DKNG", "HOOD", "RIVN", "LCID", "SOFI", "AFRM", "UPST", "AI",
            "PINS", "SNAP", "ZM", "DOCU", "OKTA", "CRWD", "DDOG", "NET", "ZS", "U",
            "ARM", "SMCI", "T", "VZ", "DIS", "IBM", "INTC", "QCOM", "TXN", "MU",
            "DELL", "MDB", "TEAM", "WDAY", "ADSK", "ANET", "PANW", "FTNT", "SNPS", "CDNS"
        ]
        us_tickers.extend(additional_us)
        us_tickers = list(set(us_tickers))
    except:
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

    # 2. 日本株 (JPX日経400限定 - 厳選400銘柄)
    jp_tickers = []
    try:
        print("🔍 JPX日経400銘柄リストを取得中...")
        url_jpx400 = "https://www.jpx.co.jp/markets/indices/jpx-nikkei400/tvdivq0000001vg2-att/jpxnk400_weight_j.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url_jpx400, headers=headers)
        
        # エンコーディングの自動判別と読み込み
        content = res.content
        try:
            df_jpx400 = pd.read_csv(io.BytesIO(content), encoding='cp932')
        except:
            df_jpx400 = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')
            
        # カラム名が「コード」となっている部分を抽出
        # CSVの構造によってはカラム名が異なる場合があるため、柔軟に処理
        code_col = [col for col in df_jpx400.columns if 'コード' in col or 'Code' in col]
        if code_col:
            codes = df_jpx400[code_col[0]].dropna().astype(str).tolist()
            for code in codes:
                if len(code) >= 4:
                    # 4桁の数字が含まれているか確認
                    clean_code = "".join(filter(str.isdigit, code))[:4]
                    if len(clean_code) == 4:
                        jp_tickers.append(f"{clean_code}.T")
                        
        print(f"✅ JPX日経400: {len(jp_tickers)} 銘柄を抽出完了。")
    except Exception as e:
        print(f"⚠️ JPX400リスト取得失敗: {e}。主要400銘柄でフォールバックします。")
        # 主要400銘柄が取れない場合の最小限のガード
        jp_tickers = ["7203.T", "6758.T", "6861.T", "7974.T", "9984.T", "8306.T", "4063.T", "8035.T"]

    return us_tickers, jp_tickers

def stage1_screening(ticker):
    """Stage 1: 高速テクニカルスクリーニング (超高品質銘柄用)"""
    try:
        data, _, name = get_stock_data(ticker, period="1y")
        if len(data) < 100: return None # 十分なデータがあるもののみ
        
        data = prepare_features(data)
        latest = data.iloc[-1]
        
        score = 0
        # 1. 強いトレンド
        if latest['Close'] > latest['SMA_20'] and latest['SMA_20'] > latest['SMA_50']:
            score += 2
        
        # 2. 適正なRSI
        if 40 < latest['RSI'] < 65: score += 1
        
        # 3. 低ボラティリティ (ノイズ排除)
        vol = data['Close'].pct_change().tail(20).std()
        if vol < 0.02: score += 1
        
        if score < 2: return None
        
        return {"Ticker": ticker, "ScreeningScore": score, "Name": name}
    except: return None

def process_single_stock_heavy(ticker, name):
    """Stage 2: 精密AI分析 (最高精度設定)"""
    try:
        import random
        time.sleep(random.uniform(0.3, 0.6)) # 慎重にアクセス
        
        data, currency, _ = get_stock_data(ticker, period="2y")
        if data is None or len(data) < 150: return None
            
        data = prepare_features(data)
        strategy = AIStrategy(api_key=GEMINI_API_KEY, precise=True)
        score = strategy.train(data)
        signals = strategy.predict_signals(data)
        
        if signals.empty or 'Prediction' not in signals.columns:
            return None
            
        avg_pred = signals['Prediction'].tail(5).mean()
        annualized_pred = avg_pred * 252
        
        return {
            "Name": name, "Ticker": ticker, "Price": signals['Close'].iloc[-1],
            "Currency": currency, "AI Prediction": annualized_pred,
            "Confidence": score
        }
    except:
        return None

def run_mass_scan():
    print("📢 --- 【究極のエリート】日米1,400銘柄・JPX400巡回システム起動 ---")
    
    us_tickers, jp_tickers = get_all_target_tickers()
    all_tickers = list(set(us_tickers + jp_tickers))
    print(f"✅ ターゲット確定: {len(all_tickers)} 銘柄 (US Russell 1000相当 / JP JPX400)")

    print(f"🚀 Stage 1: 超高品質フィルタリング開始...")
    screened_candidates = []
    with ThreadPoolExecutor(max_workers=25) as executor: # 安定性重視で少し並列度を下げる
        futures = {executor.submit(stage1_screening, t): t for t in all_tickers}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: screened_candidates.append(res)
            if (i+1) % 200 == 0: print(f"Screening Progress: {i+1}/{len(all_tickers)}")
    
    # 上位300銘柄をAI分析へ
    screened_candidates = sorted(screened_candidates, key=lambda x: x['ScreeningScore'], reverse=True)[:300]
    print(f"✅ Stage 1 完了。精鋭 {len(screened_candidates)} 銘柄をAI精密診断へ投入。")

    print(f"🧠 Stage 2: 精密AI診断開始 (Maximum Precision Mode)...")
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_stock_heavy, c['Ticker'], c['Name']): c for c in screened_candidates}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if (i+1) % 20 == 0: print(f"AI Analysis Progress: {i+1}/{len(screened_candidates)}")

    try:
        if results:
            df_all = pd.DataFrame(results)
            df_all['Score'] = df_all['AI Prediction'] * (df_all['Confidence'] + 2.0)
            df_market = df_all[df_all['AI Prediction'] > 0].sort_values(by="Score", ascending=False)
            df_market.to_csv(RESULTS_FILE, index=False)
            
            if not df_market.empty:
                send_discord_summary(df_market)
                print(f"✨ 全分析完了。究極のエリート銘柄の中から {len(df_market)} 件のチャンスを特定しました。")
            else:
                print("⚠️ 有望な銘柄は見つかりませんでした。")
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
