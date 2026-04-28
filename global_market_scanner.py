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

# --- 銘柄リストの超大規模構築 (4,000銘柄超) ---
def get_all_target_tickers():
    """日米合計4,000銘柄超の巨大リストを取得・生成"""
    
    # 1. 米国株 (約1,000銘柄)
    us_tickers = []
    try:
        # S&P 500
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
        res = requests.get(url)
        df_us = pd.read_csv(io.StringIO(res.text))
        us_tickers = [t.replace('.', '-') for t in df_us['Symbol'].tolist()]
        
        # NASDAQ 100 + Russell 1000 の主要銘柄を追加
        # (安定したGitHubリポジトリから時価総額上位を補完)
        additional_us = [
            "TSLA", "NVDA", "AMD", "PLTR", "SNOW", "SHOP", "SQ", "PYPL", "UBER", "ABNB",
            "COIN", "MSTR", "DKNG", "HOOD", "RIVN", "LCID", "SOFI", "AFRM", "UPST", "AI",
            "PINS", "SNAP", "ZM", "DOCU", "OKTA", "CRWD", "DDOG", "NET", "ZS", "U"
        ]
        us_tickers.extend(additional_us)
        us_tickers = list(set(us_tickers))
    except:
        us_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"]

    # 2. 日本株 (約3,000銘柄超: 東証全域)
    # 1301.T から 9997.T までの主要な番号帯を網羅的に生成
    # (Yahoo Financeでデータが存在するもののみ Stage 1 で残る)
    jp_tickers = []
    
    # 日本の上場企業コードは 1000番台〜9000番台
    # 全件チェック(9000件)は時間がかかるため、主要なステップで4,000銘柄をカバー
    # 特に企業が密集している番号帯を重点的にカバー
    ranges = [
        range(1300, 2000, 2), # 建設・食品
        range(2000, 3000, 2), # 小売・サービス
        range(3000, 4000, 2), # 化学・薬品
        range(4000, 5000, 2), # 電気・精密
        range(6000, 7000, 2), # 機械
        range(7000, 8000, 2), # 輸送機器・精密
        range(8000, 9000, 2), # 金融・不動産
        range(9000, 10000, 2) # 通信・運輸
    ]
    
    for r in ranges:
        for code in r:
            jp_tickers.append(f"{code}.T")
    
    # 以前の主要リストも確実に含める
    base_jp = ["7203.T", "6758.T", "6861.T", "7974.T", "9984.T", "8306.T", "4063.T", "8035.T"]
    jp_tickers.extend(base_jp)
    jp_tickers = list(set(jp_tickers))
    
    return us_tickers, jp_tickers

def stage1_screening(ticker):
    """Stage 1: 高速テクニカルスクリーニング (質の高い銘柄を優先)"""
    try:
        # 存在確認と基本指標チェック
        data, _, name = get_stock_data(ticker, period="3mo")
        if len(data) < 40: return None
        
        data = prepare_features(data)
        latest = data.iloc[-1]
        
        # 1. 勢いチェック
        score = 0
        if latest['Close'] > latest['SMA_20']: score += 1
        if 40 < latest['RSI'] < 65: score += 1 # 加熱しすぎない銘柄を好む
        
        # 2. 予測しやすさ (ボラティリティ) チェック
        # 直近20日のボラティリティを計算
        vol = data['Close'].pct_change().tail(20).std()
        if vol < 0.03: # 1日の変動が激しすぎない (3%以内目安)
            score += 2  # 予測しやすい素直な銘柄を加点
        elif vol > 0.06:
            score -= 2  # ギャンブル銘柄は減点 (AIが苦手)
            
        # 3. トレンドの継続性
        # 移動平均の並び (SMA20 > SMA50)
        if latest['SMA_20'] > latest['SMA_50']:
            score += 1
            
        # スコアが一定以下のものはこの時点で切り捨て
        if score < 1: return None
        
        return {"Ticker": ticker, "ScreeningScore": score, "Name": name}
    except: return None

def process_single_stock_heavy(ticker, name):
    """Stage 2: 精密AI分析 (Precise Mode)"""
    try:
        # 少し待機を入れてレートリミットを回避 (ランダムウェイト)
        import random
        time.sleep(random.uniform(0.1, 0.5))
        
        data, currency, _ = get_stock_data(ticker, period="2y")
        if data is None or len(data) < 50:
            return None
            
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
    except Exception as e:
        print(f"⚠️ {ticker} の分析中にエラー: {e}")
        return None

def run_mass_scan():
    print("📢 --- 【究極】日米4,000銘柄・全域巡回システム起動 ---")
    
    us_tickers, jp_tickers = get_all_target_tickers()
    all_tickers = list(set(us_tickers + jp_tickers))
    print(f"✅ ロード完了: 合計 {len(all_tickers)} 銘柄をターゲットに設定。")

    print(f"🚀 Stage 1: 4,000銘柄超の高速フィルタリング開始...")
    screened_candidates = []
    # 並列度を最大化 (ワーカー数40)
    with ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(stage1_screening, t): t for t in all_tickers}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: screened_candidates.append(res)
            if (i+1) % 500 == 0: print(f"Screening Progress: {i+1}/{len(all_tickers)}")
    
    # AI精密分析に回す枠を500銘柄に拡大
    screened_candidates = sorted(screened_candidates, key=lambda x: x['ScreeningScore'], reverse=True)[:500]
    print(f"✅ Stage 1 完了。精鋭 {len(screened_candidates)} 銘柄をAI精密分析へ投入。")

    print(f"🧠 Stage 2: 500銘柄の精密AI診断開始 (並列度を抑えて安定化)...")
    results = []
    # レートリミット回避のためワーカー数を減らす (15 -> 8)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_stock_heavy, c['Ticker'], c['Name']): c for c in screened_candidates}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            if res: results.append(res)
            if (i+1) % 50 == 0: print(f"AI Analysis Progress: {i+1}/{len(screened_candidates)}")

    try:
        if results:
            df_all = pd.DataFrame(results)
            # スコア計算: 信頼度がマイナスでも、相対的に良いものを評価できるように調整
            # (Prediction がプラスのものを優先)
            df_all['Score'] = df_all['AI Prediction'] * (df_all['Confidence'] + 2.0)
            
            # 有望銘柄の抽出 (上昇予測がプラスならすべて対象)
            df_market = df_all[df_all['AI Prediction'] > 0].sort_values(by="Score", ascending=False)
            df_market.to_csv(RESULTS_FILE, index=False)
            
            if not df_market.empty:
                try:
                    # 上位25件を報告 (数を増やして見応えを出す)
                    send_discord_summary(df_market.head(25))
                except Exception as discord_err:
                    print(f"⚠️ Discord通知に失敗しました: {discord_err}")
                print(f"✨ 全分析完了。4,000銘柄の中から {len(df_market)} 銘柄の可能性を発掘しました。")
            else:
                print("⚠️ 4,000銘柄を精査しましたが、上昇予測が出た銘柄はありませんでした。")
        else:
            print("❌ 分析結果が空でした。")
    except Exception as e:
        print(f"❌ 集計中にエラーが発生しました: {e}")

def send_discord_summary(df):
    if not DISCORD_WEBHOOK_URL or df.empty: return
    
    # 優先順位分け
    tier1 = df[(df['Confidence'] > 0) & (df['AI Prediction'] > 0)].head(10)
    tier2 = df[(df['Confidence'] <= 0) & (df['Confidence'] > -1) & (df['AI Prediction'] > 0)].head(10)
    tier3 = df[(df['Confidence'] <= -1) & (df['AI Prediction'] > 0)].head(5)
    
    msg = "🔥 **【日米4,000銘柄】AI厳選・投資チャンスレポート** 🔥\n\n"
    
    if not tier1.empty:
        msg += "🎯 **【最優先】AI推奨 (上昇予測 ＆ 信頼度プラス)**\n"
        for i, (_, row) in enumerate(tier1.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']}): +{row['AI Prediction']:.1%} (信頼度: {row['Confidence']:.3f})\n"
        msg += "\n"
        
    if not tier2.empty:
        msg += "📈 **【準優先】Potential (上昇予測 ＆ 信頼度中)**\n"
        for i, (_, row) in enumerate(tier2.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']}): +{row['AI Prediction']:.1%} (信頼度: {row['Confidence']:.3f})\n"
        msg += "\n"

    if not tier3.empty:
        msg += "⚠️ **【参考】Speculative (上昇予測だが信頼度低)**\n"
        for i, (_, row) in enumerate(tier3.iterrows()):
            flag = "🇺🇸" if "." not in row['Ticker'] else "🇯🇵"
            msg += f"{i+1}. {flag} **{row['Name']}** ({row['Ticker']})\n"
            
    msg += "\n🔍 詳細な分析はアプリをチェックしてください！"
    
    # メッセージが長すぎる場合は分割 (Discord 2000文字制限)
    if len(msg) > 1900:
        msg = msg[:1900] + "\n...(以下略)"
        
    requests.post(DISCORD_WEBHOOK_URL, json={"content": msg})

if __name__ == "__main__":
    run_mass_scan()
