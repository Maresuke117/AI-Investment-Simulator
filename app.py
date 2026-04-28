import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from yahooquery import Ticker
from data_loader import get_stock_data, search_ticker
from strategy import AIStrategy, prepare_features
from dotenv import load_dotenv
import os
import time

# .envファイルを読み込む
load_dotenv()

st.set_page_config(page_title="AI Stock Trader", layout="wide")

st.title("🚀 AI Dynamic Stock Trading System")
st.markdown("AIを活用した個別株式の動的トレードシミュレーター")

# ファイルパスの絶対パス化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "market_scan_results.csv")
PORTFOLIO_RESULTS_FILE = os.path.join(BASE_DIR, "portfolio_scan_results.csv")
REPORT_CACHE_FILE = os.path.join(BASE_DIR, "reports_cache.json")

import json

def load_report_cache():
    if os.path.exists(REPORT_CACHE_FILE):
        try:
            with open(REPORT_CACHE_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {}

def save_report_cache(cache):
    with open(REPORT_CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False)

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return json.load(f)
        except: pass
    return {"total_budget": 1000000, "target_return": 30, "period": "2y", "display_currency": "JPY"}

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            df = pd.read_csv(PORTFOLIO_FILE)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Portfolio load error: {e}")
    return pd.DataFrame(columns=["Ticker", "Buy Price", "Quantity"])

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

@st.cache_data(ttl=3600)
def get_exchange_rate():
    """USD/JPYの為替レートを取得"""
    try:
        t = Ticker("USDJPY=X")
        data = t.history(period="1d")
        if 'close' in data.columns:
            return data['close'].iloc[-1]
        elif 'Close' in data.columns:
            return data['Close'].iloc[-1]
        return 150.0
    except:
        return 150.0 # 取得失敗時のデフォルト

# --- 設定とデータの読み込み ---
user_settings = load_settings()
usdjpy = get_exchange_rate()

# サイドバー設定
st.sidebar.header("Settings")
options_period = ["1y", "2y", "5y", "max"]
default_period_idx = options_period.index(user_settings.get("period", "2y"))
new_period = st.sidebar.selectbox("Period", options=options_period, index=default_period_idx)

# 表示通貨の設定
display_currency = st.sidebar.selectbox("Display Currency (表示通貨)", options=["JPY", "USD"], index=0 if user_settings.get("display_currency") == "JPY" else 1)

# 為替レートの表示
st.sidebar.metric("Current USD/JPY", f"¥{usdjpy:.2f}")

# AI Engine Settings
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Settings")

azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if azure_key and azure_endpoint and azure_deployment:
    st.sidebar.success(f"✅ Azure OpenAI Active\n({azure_deployment})")
    api_key = None
else:
    default_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")
    if not api_key:
        st.sidebar.info("💡 Azure OpenAI未設定のためGeminiキーが必要です")

# 投資予算設定
st.sidebar.markdown("---")
st.sidebar.header("Money Management")
new_budget = st.sidebar.number_input(f"Total Investment Budget ({display_currency})", min_value=10000, value=user_settings["total_budget"], step=100000)
new_target = st.sidebar.slider("Target Annual Return (%)", min_value=10, max_value=100, value=user_settings["target_return"])

# 設定が変更されたら保存
if (new_budget != user_settings["total_budget"] or 
    new_target != user_settings["target_return"] or 
    new_period != user_settings.get("period") or
    display_currency != user_settings.get("display_currency")):
    save_settings({
        "total_budget": new_budget, 
        "target_return": new_target, 
        "period": new_period,
        "display_currency": display_currency
    })

total_budget = new_budget
target_return_filter = new_target
period = new_period

tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Stock Screener", "💰 My Portfolio"])

with tab1:
    # (既存のコードと同じ)
    st.subheader("Individual Stock Analysis")
    ticker_input = st.text_input("Ticker Symbol or Company Name (例: 7203, トヨタ, AAPL)", value="AAPL", key="single_ticker")
    if st.button("Run Analysis", key="btn_single"):
        try:
            # 検索ロジック: 4桁数字でもティッカーシンボルでもない場合は検索を試みる
            target_ticker = ticker_input
            if not (target_ticker.isdigit() and len(target_ticker) == 4) and not (target_ticker.isupper() and target_ticker.isalpha()):
                with st.spinner(f"Searching for '{ticker_input}'..."):
                    found = search_ticker(ticker_input)
                    if found:
                        target_ticker = found
                        st.success(f"Found: {target_ticker}")
                    else:
                        st.error(f"'{ticker_input}' に該当する銘柄が見つかりませんでした。正しいティッカーコード（例: 7203, AAPL）を直接入力してください。")
                        st.stop() # 処理を中断
            
            with st.spinner("Fetching data..."):
                data, currency = get_stock_data(target_ticker, period=period)
            
            strategy = AIStrategy(api_key=api_key)
            score = strategy.train(data)
            signals = strategy.predict_signals(data)
            
            currency_symbol = "¥" if currency == "JPY" else "$"
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Model R^2 Score", f"{score:.4f}", 
                        help="AIが将来の収益率をどの程度正確に予測できているかを示す指標（最大1.0）。0.1以上なら高い信頼性、マイナスは予測が困難な状態を意味します。")
            
            signals['Strategy_Return'] = signals['Signal'].shift(1) * signals['Target']
            cumulative_market = (1 + signals['Target']).cumprod()
            cumulative_strategy = (1 + signals['Strategy_Return']).cumprod()
            
            total_return = (cumulative_strategy.iloc[-1] - 1) * 100
            col2.metric("Total Strategy Return", f"{total_return:.2f}%",
                        help="AIのシグナル（買い・売り）に従って売買を繰り返した場合の累積収益率です。")
            col3.metric("Market Return", f"{(cumulative_market.iloc[-1] - 1) * 100:.2f}%",
                        help="同じ期間、その銘柄を売買せずに持ち続けた場合（バイ・アンド・ホールド）の収益率です。")

            st.subheader(f"Price & Signals: {target_ticker} ({currency})")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], name=f'Close Price ({currency_symbol})', line=dict(color='white', width=2)))
            buy_signals = signals[signals['Signal'] == 1]
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', 
                                     name='Buy Signal', marker=dict(symbol='triangle-up', size=10, color='lime')))
            st.plotly_chart(fig, use_container_x_supported=True)

            # Geminiによるセンチメント分析
            if api_key:
                st.subheader("📰 AI Sentiment Analysis (Gemini Pro)")
                with st.spinner("Analyzing recent news..."):
                    try:
                        # yahooqueryからニュースを取得
                        stock = Ticker(target_ticker)
                        news = stock.news
                        
                        if news and isinstance(news, list):
                            news_items = []
                            for n in news[:8]:
                                title = n.get('title')
                                summary = n.get('summary', "")
                                provider = n.get('publisher', 'Unknown')
                                
                                if title:
                                    news_items.append(f"- {title} ({provider})\n  要約: {summary[:100]}...")
                            
                            if news_items:
                                news_text = "\n".join(news_items)
                                sentiment_json_str = strategy.get_sentiment(target_ticker, news_text)
                                
                                try:
                                    import json
                                    # JSON部分を抽出 (稀にGeminiがMarkdownのデコレーションを付けるため)
                                    clean_json = sentiment_json_str.strip()
                                    if "```json" in clean_json:
                                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                                    elif "```" in clean_json:
                                        clean_json = clean_json.split("```")[1].split("```")[0].strip()
                                    
                                    res = json.loads(clean_json)
                                    score = res.get("score", 0)
                                    reason = res.get("reason", "")
                                    
                                    # 視覚的な表示
                                    col_s1, col_s2 = st.columns([1, 4])
                                    
                                    # スコアに応じた判定とアクション
                                    if score >= 0.7:
                                        status, action, color = "超強気 (Very Bullish)", "積極的な買付を検討", "lime"
                                    elif score >= 0.2:
                                        status, action, color = "強気 (Bullish)", "買付・ホールドが有利", "green"
                                    elif score <= -0.7:
                                        status, action, color = "超弱気 (Very Bearish)", "即時売却・回避を推奨", "red"
                                    elif score <= -0.2:
                                        status, action, color = "弱気 (Bearish)", "警戒・一部利確を検討", "orange"
                                    else:
                                        status, action, color = "中立 (Neutral)", "現状維持・様子見", "white"
                                    
                                    col_s1.metric("Sentiment Score", f"{score:+.2f}", delta=status, delta_color="normal")
                                    st.markdown(f"**AI推奨アクション:** :{color}[{action}]")
                                    st.markdown(f"**AI判断理由:**\n{reason}")
                                    
                                    with st.expander("ℹ️ スコアの読み方"):
                                        st.write("""
                                        - **+0.7以上**: 非常に強い材料。積極的な投資チャンス。
                                        - **+0.2〜+0.7**: 良好な材料。買い・ホールドに値する。
                                        - **-0.2〜+0.2**: 材料が乏しい、または賛否両論。静観。
                                        - **-0.2〜-0.7**: 懸念材料あり。リスク管理を優先。
                                        - **-0.7以下**: 深刻な悪材料。撤退や回避を強く検討。
                                        """)
                                except Exception as e:
                                    st.info(sentiment_json_str) # パース失敗時はそのまま表示
                            else:
                                st.warning(f"'{target_ticker}' に関する解析可能なニュースが見つかりませんでした。")
                                with st.expander("生データを確認 (デバッグ用)"):
                                    st.write(news[:1])
                        else:
                            st.write(f"'{target_ticker}' に関するニュースが配信されていません。")
                    except Exception as e:
                        st.warning(f"Sentiment analysis failed: {e}")
            else:
                st.info("💡 サイドバーにGemini APIキーを入力すると、AIによるニュース分析が有効になります。")

        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("🚀 Global Market Mass Screener")
    
    # 自動スキャン結果の表示セクション
    if os.path.exists(RESULTS_FILE):
        with st.expander("📅 最新の自動スキャン結果を表示 (AI推薦銘柄ランキング)", expanded=True):
            try:
                df_auto = pd.read_csv(RESULTS_FILE)
                if 'Confidence' not in df_auto.columns:
                    df_auto['Confidence'] = 0.0
                st.write(f"最終更新: {time.ctime(os.path.getmtime(RESULTS_FILE))}")
                
                # 通貨記号の付与
                df_auto['Price_Display'] = df_auto.apply(lambda x: f"{'¥' if x['Currency']=='JPY' else '$'}{x['Price']:,.1f}", axis=1)
                
                # データのフィルタリングと分類
                # 1. 予測がマイナスのものは除外
                df_all_plus = df_auto[df_auto['AI Prediction'] > 0].copy()
                
                # 2. メイン (Confidence > -1.0)
                df_main = df_all_plus[df_all_plus['Confidence'] > -1.0].copy()
                # スコア計算 (ソート用)
                df_main['Score'] = df_main['AI Prediction'] * np.where(df_main['Confidence'] > 0, df_main['Confidence'], 0.001)
                df_main = df_main.sort_values(by="Score", ascending=False)
                
                # 3. 別枠 (Confidence <= -1.0)
                df_sub = df_all_plus[df_all_plus['Confidence'] <= -1.0].copy()
                df_sub = df_sub.sort_values(by="Confidence", ascending=False)
                
                # --- メインセクション ---
                st.markdown("### 🎯 AI Recommended (上昇予測 ＆ 信頼度 -1.0超)")
                if not df_main.empty:
                    st.info(f"📋 **メイン銘柄 Ticker 一括コピー ({len(df_main)}件)**")
                    st.code(",".join(df_main['Ticker'].tolist()))
                    st.table(df_main[["Name", "Ticker", "Price_Display", "AI Prediction", "Confidence"]].style.format({
                        "AI Prediction": "{:.2%}",
                        "Confidence": "{:.4f}"
                    }))
                else:
                    st.warning("メイン条件を満たす銘柄はありません。")
                
                # --- 別枠セクション (折りたたみ) ---
                with st.expander("⚠️ 参考：上昇予測だが信頼度が極めて低い銘柄 (Confidence ≦ -1.0)"):
                    st.write("AIは上昇を予測していますが、モデルの信頼度が極めて低いため注意が必要です。")
                    if not df_sub.empty:
                        st.table(df_sub[["Name", "Ticker", "Price_Display", "AI Prediction", "Confidence"]].style.format({
                            "AI Prediction": "{:.2%}",
                            "Confidence": "{:.4f}"
                        }))
                    else:
                        st.write("該当なし")
                
                # 自動スキャンの上位5件に対して詳細な投資戦略を生成
                st.subheader("💡 AI Recommended Investment Strategy (Top 5)")
                report_strategy = AIStrategy(api_key=api_key)
                report_cache = load_report_cache()
                
                for _, row in df_auto.head(5).iterrows():
                    ticker = row['Ticker']
                    cache_key = f"{ticker}_{row['Price']:.1f}"
                    
                    with st.expander(f"💰 {ticker} の投資戦略レポート (自動診断)"):
                        if cache_key in report_cache:
                            st.markdown(report_cache[cache_key])
                        else:
                            with st.spinner(f"Generating strategy for {ticker}..."):
                                try:
                                    # ニュースを取得してセンチメントを計算
                                    stock = yf.Ticker(ticker)
                                    news = stock.news
                                    s_score = 0
                                    if news:
                                        news_text = "\n".join([n.get('content', {}).get('title', '') for n in news[:5]])
                                        try:
                                            import json
                                            raw_s = report_strategy.get_sentiment(ticker, news_text)
                                            clean_json = raw_s.strip()
                                            if "```json" in clean_json:
                                                clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                                            res = json.loads(clean_json)
                                            s_score = res.get("score", 0)
                                        except: pass
                                    
                                    native_currency = "JPY" if row['Currency']=='JPY' else "USD"
                                    effective_budget = total_budget
                                    if display_currency == "JPY" and native_currency == "USD": effective_budget = total_budget / usdjpy
                                    elif display_currency == "USD" and native_currency == "JPY": effective_budget = total_budget * usdjpy
                                    
                                    c_symbol = "¥" if native_currency == "JPY" else "$"
                                    advice = report_strategy.get_investment_advice(ticker, row['Price'], row['AI Prediction'], s_score, effective_budget, c_symbol)
                                    
                                    # キャッシュに保存
                                    report_cache[cache_key] = advice
                                    save_report_cache(report_cache)
                                    
                                    st.markdown(advice)
                                except Exception as e:
                                    st.warning(f"レポート生成に失敗しました: {e}")

                st.info("💡 毎日深夜に自動実行されるAIスキャンの結果です。")
            except Exception as e:
                st.error(f"スキャン結果の読み込みに失敗しました: {e}")
    
    st.markdown("---")
    st.subheader("🛠 カスタム・一括スキャナー")
    st.markdown("特定の銘柄群を今すぐ手動でスキャンしたい場合に使用してください。")
    
    # プリセット銘柄の定義
    PRESETS = {
        "カスタム入力": "",
        "🇺🇸 S&P 100 (米国時価総額上位)": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,GOOG,BRK-B,TSLA,UNH,LLY,AVGO,XOM,JPM,V,MA,JNJ,PG,COST,HD,ABBV,ADBE,CRM,WMT,BAC,CVX,ACN,PEP,MRK,KO,LIN,AMD,TMO,CSCO,ORCL,MCD,INTC,ABT,GE,CAT,VZ,DIS,DHR,INTU,CMCSA,PM,PFE,IBM,AMAT,QCOM,UNP,ISRG,LOW,HON,TXN,SPGI,AMGN,AXP,COP,MS,RTX,NEE,GS,ELV,SCHW,SYK,LMT,TJX,BLK,DE,ETN,UPS,PGR,C,REGN,SBUX,MDLZ,VRTX,MMC,CB,BSX,PLD,ADI,PANW,BA,CI,GILD,FI,SNPS,BMY,MU,HCA,CDNS,T,CRWD,LRCX,MAR,SHW,ITW",
        "🇯🇵 日経平均採用 (主要100銘柄)": "7203.T,6758.T,6861.T,7974.T,9984.T,8306.T,4063.T,8035.T,6857.T,6920.T,6367.T,4502.T,4503.T,6501.T,8001.T,8031.T,8058.T,9432.T,9433.T,2914.T,3382.T,4519.T,4568.T,4661.T,4901.T,6098.T,6146.T,6273.T,6503.T,6506.T,6594.T,6701.T,6702.T,6723.T,6902.T,6954.T,6981.T,7011.T,7267.T,7309.T,7733.T,7741.T,7832.T,8002.T,8113.T,8267.T,8316.T,8411.T,8591.T,8604.T,8630.T,8725.T,8766.T,8801.T,8802.T,9020.T,9021.T,9022.T,9101.T,9104.T,9107.T,9201.T,9202.T,9501.T,9502.T,9503.T,9613.T,9735.T,9843.T,9983.T",
        "💻 半導体・AI関連 (日米)": "NVDA,ASML,TSM,AMD,ARM,8035.T,6857.T,6146.T,PLTR,SMCI,AVGO,6723.T"
    }
    
    # 選択が変更された時にテキストエリアを更新する関数
    def update_presets():
        choice = st.session_state.preset_select_mass
        if choice in PRESETS and PRESETS[choice]:
            st.session_state.ticker_input_mass = PRESETS[choice]

    # プルダウン（変更時にupdate_presetsを呼ぶ）
    preset_choice = st.selectbox(
        "カテゴリーから一括選択", 
        options=list(PRESETS.keys()), 
        key="preset_select_mass",
        on_change=update_presets
    )
    
    # 初期値の設定
    if "ticker_input_mass" not in st.session_state:
        st.session_state.ticker_input_mass = "AAPL,MSFT,NVDA"

    ticker_list_raw = st.text_area(
        "対象ティッカー", 
        key="ticker_input_mass"
    )
    
    CUSTOM_RESULTS_FILE = "custom_scan_results.csv"
    
    use_parallel = st.checkbox("高速並列スキャンを使用 (推奨)", value=True, key="use_parallel_mass")
    
    col_btn1, col_btn2 = st.columns([1, 4])
    start_btn = col_btn1.button("🚀 大量スキャニング開始", key="btn_screen_mass")
    
    if os.path.exists(CUSTOM_RESULTS_FILE):
        if col_btn2.button("🗑️ 結果をリセット", key="btn_reset_mass"):
            os.remove(CUSTOM_RESULTS_FILE)
            st.rerun()

    if start_btn:
        tickers = [t.strip() for t in ticker_list_raw.split(",") if t.strip()]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_ticker(t_input):
            try:
                target_t = t_input
                if target_t.isdigit() and len(target_t) == 4: target_t += ".T"
                data, currency = get_stock_data(target_t, period=period)
                data = prepare_features(data)
                if len(data) < 50: return None
                
                strategy = AIStrategy(api_key=api_key)
                strategy.train(data)
                signals = strategy.predict_signals(data)
                
                avg_pred = signals['Prediction'].tail(5).mean()
                annualized_pred = avg_pred * 252
                
                try:
                    name = yf.Ticker(target_t).info.get('shortName', target_t)
                except:
                    name = target_t

                return {
                    "Name": name,
                    "Ticker": target_t,
                    "Price": signals['Close'].iloc[-1],
                    "Currency": currency,
                    "AI Prediction": annualized_pred,
                    "Confidence": strategy.train(data)
                }
            except:
                return None

        if use_parallel:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
                for i, future in enumerate(as_completed(future_to_ticker)):
                    res = future.result()
                    if res: results.append(res)
                    progress_bar.progress((i + 1) / len(tickers))
        else:
            for i, t in enumerate(tickers):
                res = process_ticker(t)
                if res: results.append(res)
                progress_bar.progress((i + 1) / len(tickers))

        if results:
            df_res = pd.DataFrame(results)
            df_res.to_csv(CUSTOM_RESULTS_FILE, index=False)
            st.rerun()
    # 保存された結果がある場合は表示
    if os.path.exists(CUSTOM_RESULTS_FILE):
        try:
            df_res = pd.read_csv(CUSTOM_RESULTS_FILE)
            if 'Confidence' not in df_res.columns:
                df_res['Confidence'] = 0.0
            df_res['Price_Display'] = df_res.apply(lambda x: f"{'¥' if x['Currency']=='JPY' else '$'}{x['Price']:,.1f}", axis=1)
            
            # 1. 予測がマイナスのものは除外
            df_all_plus = df_res[df_res['AI Prediction'] > 0].copy()
            
            # 2. メイン (Confidence > -1.0)
            df_main = df_all_plus[df_all_plus['Confidence'] > -1.0].copy()
            # スコア計算 (ソート用: Confidenceが正なら掛け算、負なら最小限の重み)
            df_main['Score'] = df_main['AI Prediction'] * np.where(df_main['Confidence'] > 0, df_main['Confidence'], 0.001)
            df_main = df_main.sort_values(by="Score", ascending=False)
            
            # 3. 別枠 (Confidence <= -1.0)
            df_sub = df_all_plus[df_all_plus['Confidence'] <= -1.0].copy()
            df_sub = df_sub.sort_values(by="Confidence", ascending=False)

            st.write(f"### 🎯 分析結果 ({len(df_res)}件中、上昇予測 {len(df_all_plus)}件)")
            
            # --- メインセクション ---
            st.markdown("#### 🌟 AI Recommended (上昇予測 ＆ 信頼度 -1.0超)")
            if not df_main.empty:
                st.info(f"📋 **メイン銘柄 Ticker 一括コピー ({len(df_main)}件)**")
                st.code(",".join(df_main['Ticker'].tolist()))
                st.table(df_main[["Name", "Ticker", "Price_Display", "AI Prediction", "Confidence"]].style.format({
                    "AI Prediction": "{:.2%}",
                    "Confidence": "{:.4f}"
                }))
            else:
                st.warning("メイン条件を満たす銘柄はありませんでした。")
            
            # --- 別枠セクション ---
            with st.expander("⚠️ 参考：上昇予測だが信頼度が極めて低い銘柄 (Confidence ≦ -1.0)"):
                if not df_sub.empty:
                    st.table(df_sub[["Name", "Ticker", "Price_Display", "AI Prediction", "Confidence"]].style.format({
                        "AI Prediction": "{:.2%}",
                        "Confidence": "{:.4f}"
                    }))
                else:
                    st.write("該当なし")
            
            # Ticker一括コピー機能の追加
            st.write("📋 **Ticker 一括コピー用 (上位50件)**")
            st.code(",".join(df_res['Ticker'].head(50).tolist()))
            
            # 上位5件に対して詳細な投資戦略を生成
            st.subheader("💡 AI Recommended Investment Strategy (Top 5)")
            report_strategy = AIStrategy(api_key=api_key)
            report_cache = load_report_cache()
            
            for _, row in df_res.head(5).iterrows():
                ticker = row['Ticker']
                cache_key = f"{ticker}_{row['Price']:.1f}"
                
                with st.expander(f"💰 {ticker} の投資戦略レポート"):
                    if cache_key in report_cache:
                        st.markdown(report_cache[cache_key])
                    else:
                        with st.spinner(f"Generating strategy for {ticker}..."):
                            try:
                                # ニュースを取得してセンチメントを計算
                                t_obj = Ticker(ticker)
                                news = t_obj.news
                                s_score = 0
                                if news and isinstance(news, list):
                                    news_text = "\n".join([n.get('title', '') for n in news[:5]])
                                    try:
                                        import json
                                        raw_s = report_strategy.get_sentiment(ticker, news_text)
                                        clean_json = raw_s.strip()
                                        if "```json" in clean_json:
                                            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                                        res = json.loads(clean_json)
                                        s_score = res.get("score", 0)
                                    except: pass
                                
                                # 予算を銘柄の通貨に合わせて換算する
                                native_currency = "JPY" if row['Currency']=='JPY' else "USD"
                                effective_budget = total_budget
                                if display_currency == "JPY" and native_currency == "USD": effective_budget = total_budget / usdjpy
                                elif display_currency == "USD" and native_currency == "JPY": effective_budget = total_budget * usdjpy
                                
                                c_symbol = "¥" if native_currency == "JPY" else "$"
                                advice = report_strategy.get_investment_advice(ticker, row['Price'], row['AI Prediction'], s_score, effective_budget, c_symbol)
                                
                                # キャッシュに保存
                                report_cache[cache_key] = advice
                                save_report_cache(report_cache)
                                
                                st.markdown(advice)
                            except Exception as e:
                                st.warning(f"レポート生成に失敗しました: {e}")
        except Exception as e:
            st.error(f"分析結果の表示中にエラーが発生しました: {e}")
    else:
        st.info("💡 まだ手動分析の結果はありません。上のボタンからスキャニングを開始してください。")

with tab3:
    st.subheader("💰 My Portfolio Management")
    st.markdown("あなたの保有銘柄を登録し、AIからの売買アドバイスを確認します。")
    
    portfolio_df = load_portfolio()
    
    # 自動診断結果の表示
    if os.path.exists(PORTFOLIO_RESULTS_FILE):
        with st.expander("📅 毎朝のAI自動診断レポート", expanded=True):
            try:
                df_p_auto = pd.read_csv(PORTFOLIO_RESULTS_FILE)
                st.write(f"最終診断: {time.ctime(os.path.getmtime(PORTFOLIO_RESULTS_FILE))}")
                
                def get_status_label(pred):
                    if pred > 0.2: return "🔥 強気"
                    if pred > 0: return "✅ 堅調"
                    return "⚠️ 注意"
                
                df_p_auto['Status'] = df_p_auto['AI Prediction'].apply(get_status_label)
                st.table(df_p_auto[["Name", "Ticker", "Status", "AI Prediction"]].style.format({"AI Prediction": "{:.2%}"}))
                st.info("💡 毎朝7時に実行される自動診断の結果です。")
            except Exception as e:
                st.error(f"診断結果の読み込みに失敗しました: {e}")
    
    # 新規登録
    with st.expander("➕ 新しい持ち株を登録する"):
        st.info("💡 日本株は銘柄コード（例: 7203）を入力すると円建て、米国株（例: AAPL）はドル建てとして扱われます。")
        with st.form("add_holding_form"):
            new_ticker = st.text_input("銘柄コード (例: 7203, NVDA)", help="日本株は4桁の数字、米国株はシンボルを入力してください")
            new_price = st.number_input("取得単価", min_value=0.0, step=0.1, help="選択した通貨での単価を入力してください")
            new_currency = st.radio("通貨", options=["JPY", "USD"], horizontal=True)
            new_qty = st.number_input("数量", min_value=0, step=1)
            if st.form_submit_button("登録"):
                new_row = pd.DataFrame([{"Ticker": new_ticker, "Buy Price": new_price, "Quantity": new_qty, "Currency": new_currency}])
                portfolio_df = pd.concat([portfolio_df, new_row], ignore_index=True)
                save_portfolio(portfolio_df)
                st.success(f"{new_ticker} ({new_currency}) を登録しました！")
                st.rerun()

    if not portfolio_df.empty:
        st.subheader("現在の保有状況とAIアドバイス")
        
        # 保有銘柄の分析結果を保存するセッション状態の初期化
        if 'portfolio_analysis' not in st.session_state:
            st.session_state.portfolio_analysis = {}
        
        # 一括分析ボタン
        if st.button("🚀 全銘柄を一括AI分析", type="primary"):
            with st.status("Analyzing entire portfolio...", expanded=True) as status:
                for index, row in portfolio_df.iterrows():
                    ticker = row['Ticker']
                    status.write(f"Analyzing {ticker}...")
                    try:
                        data, _detected_currency = get_stock_data(ticker, period="2y")
                        data = prepare_features(data)
                        
                        # ユーザー指定の通貨があれば優先し、なければ形式から推測
                        if 'Currency' in row and pd.notna(row['Currency']):
                            currency = row['Currency']
                        elif ticker.endswith('.T') or (ticker.isdigit() and len(ticker) == 4):
                            currency = "JPY"
                        else:
                            currency = _detected_currency
                        
                        strategy = AIStrategy(api_key=api_key)
                        score = strategy.train(data)
                        signals = strategy.predict_signals(data)
                        
                        current_price = signals['Close'].iloc[-1]
                        prediction = signals['Prediction'].iloc[-1]
                        
                        advice, color_type = strategy.get_advice(current_price, row['Buy Price'], prediction)
                        profit_pct = (current_price / row['Buy Price'] - 1) * 100
                        
                        # 企業名取得
                        try:
                            t_obj = Ticker(ticker)
                            comp_name = t_obj.price.get(ticker, {}).get('shortName', ticker)
                        except:
                            comp_name = ticker

                        st.session_state.portfolio_analysis[ticker] = {
                            "Name": comp_name,
                            "Price": current_price,
                            "Prediction": prediction,
                            "Advice": advice,
                            "ProfitPct": profit_pct,
                            "Currency": currency,
                            "Data": data # 詳細分析用
                        }
                    except Exception as e:
                        st.error(f"{ticker} の分析に失敗: {e}")
                status.update(label="Analysis Complete!", state="complete", expanded=False)

        # 保有銘柄ごとに表示
        portfolio_results = []
        for index, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            # 通貨の取得 (CSVに保存された値を優先し、なければ形式から推測)
            if 'Currency' in row and pd.notna(row['Currency']):
                currency = row['Currency']
            elif ticker.endswith('.T') or (ticker.isdigit() and len(ticker) == 4):
                currency = "JPY"
            else:
                currency = "USD"
                
            analysis = st.session_state.portfolio_analysis.get(ticker)
            
            if analysis:
                current_price = analysis['Price']
                profit_pct = analysis['ProfitPct']
                advice = analysis['Advice']
                comp_name = analysis['Name']
                currency = analysis['Currency']
            else:
                current_price = 0
                profit_pct = 0
                advice = "未分析 (上のボタンを押してください)"
                comp_name = ticker
                currency = "USD"
            
            # 通貨換算ロジック (表示用)
            native_currency = "JPY" if currency == "JPY" else "USD"
            disp_buy = row['Buy Price']
            disp_curr = current_price
            
            if display_currency == "JPY" and native_currency == "USD":
                disp_buy *= usdjpy
                disp_curr *= usdjpy
            elif display_currency == "USD" and native_currency == "JPY":
                disp_buy /= usdjpy
                disp_curr /= usdjpy
            
            currency_symbol = "¥" if display_currency == "JPY" else "$"
            
            portfolio_results.append({
                "銘柄名": comp_name,
                "コード": ticker,
                "通貨": native_currency,
                "数量": row['Quantity'],
                f"買値 ({display_currency})": f"{currency_symbol}{disp_buy:,.1f}",
                f"現在値 ({display_currency})": f"{currency_symbol}{disp_curr:,.1f}",
                "損益": f"{profit_pct:+.2f}%" if analysis else "---",
                "AIアドバイス": advice
            })
        
        # テーブル表示
        st.table(pd.DataFrame(portfolio_results))
        
        # 個別詳細と削除機能
        for index, row in portfolio_df.iterrows():
            ticker = row['Ticker']
            analysis = st.session_state.portfolio_analysis.get(ticker)
            
            with st.expander(f"🔍 {ticker} の詳細管理"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**銘柄名:** {comp_name}")
                    st.write(f"**取得単価:** {currency_symbol}{disp_buy:,.1f} ({display_currency})")
                    if analysis:
                        # 詳細AIレポート
                        if st.button(f"📖 {ticker} の詳細AIレポートを生成", key=f"rpt_{ticker}"):
                            with st.spinner("AI Analyzing..."):
                                try:
                                    t_obj = Ticker(ticker)
                                    news = t_obj.news
                                    news_text = "\n".join([n.get('title', '') for n in news[:5]]) if news else "No news found"
                                    
                                    strategy = AIStrategy(api_key=api_key)
                                    raw_s = strategy.get_sentiment(ticker, news_text)
                                    # ... (省略: 以前のレポート生成ロジック) ...
                                    st.markdown(strategy.get_investment_advice(ticker, analysis['Price'], analysis['Prediction'], 0.5, total_budget, currency_symbol))
                                except Exception as e:
                                    st.error(f"レポート生成失敗: {e}")
                    else:
                        st.write("詳細を表示するには一括分析を実行してください。")
                
                with col2:
                    if st.button(f"🗑 削除", key=f"del_{ticker}"):
                        portfolio_df = portfolio_df.drop(index)
                        save_portfolio(portfolio_df)
                        if ticker in st.session_state.portfolio_analysis:
                            del st.session_state.portfolio_analysis[ticker]
                        st.success(f"{ticker} を削除しました。")
                        st.rerun()
            
        # 全削除ボタン
        st.markdown("---")
        if st.button("🚨 ポートフォリオを全消去"):
            if os.path.exists(PORTFOLIO_FILE):
                os.remove(PORTFOLIO_FILE)
                st.success("ポートフォリオをリセットしました。")
                st.rerun()
    else:
        st.info("保有銘柄が登録されていません。上のフォームから登録してください。")
