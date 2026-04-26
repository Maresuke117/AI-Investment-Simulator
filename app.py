import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from data_loader import get_stock_data, search_ticker
from strategy import AIStrategy, prepare_features
from dotenv import load_dotenv
import os

# .envファイルを読み込む
load_dotenv()

st.set_page_config(page_title="AI Stock Trader", layout="wide")

st.title("🚀 AI Dynamic Stock Trading System")
st.markdown("AIを活用した個別株式の動的トレードシミュレーター")

# ファイルパスの絶対パス化
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
PORTFOLIO_FILE = os.path.join(BASE_DIR, "portfolio.csv")

import json

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
        data = yf.Ticker("USDJPY=X").history(period="1d")
        return data['Close'].iloc[-1]
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

default_api_key = os.getenv("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input("Gemini API Key", value=default_api_key, type="password")

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
                        # yfinanceからニュースを取得
                        stock = yf.Ticker(target_ticker)
                        news = stock.news
                        
                        if news and isinstance(news, list):
                            news_items = []
                            for n in news[:8]:
                                # ネストされた構造 (content) からデータを取得
                                content = n.get('content', {})
                                title = content.get('title') or content.get('headline')
                                summary = content.get('summary') or ""
                                provider = content.get('provider', {}).get('displayName', 'Unknown')
                                
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
    st.subheader("🚀 High-Speed Mass Screener")
    st.markdown("数百銘柄から有望株を高速に抽出します。")
    
    # 拡張プリセット
    PRESETS = {
        "カスタム入力": "",
        "🇺🇸 S&P 100 (米国時価総額上位)": "AAPL,MSFT,AMZN,NVDA,GOOGL,META,GOOG,BRK-B,TSLA,UNH,LLY,AVGO,XOM,JPM,V,MA,JNJ,PG,COST,HD,ABBV,ADBE,CRM,WMT,BAC,CVX,ACN,PEP,MRK,KO,LIN,AMD,TMO,CSCO,ORCL,MCD,INTC,ABT,GE,CAT,VZ,DIS,DHR,INTU,CMCSA,PM,PFE,IBM,AMAT,QCOM,UNP,ISRG,LOW,HON,TXN,SPGI,AMGN,AXP,COP,MS,RTX,NEE,GS,ELV,SCHW,SYK,LMT,TJX,BLK,DE,ETN,UPS,PGR,C,REGN,SBUX,MDLZ,VRTX,MMC,CB,BSX,PLD,ADI,PANW,BA,CI,GILD,FI,SNPS,BMY,MU,HCA,CDNS,T,CRWD,LRCX,MAR,SHW,ITW",
        "🇯🇵 日経平均採用 (主要100銘柄)": "7203.T,6758.T,6861.T,7974.T,9984.T,8306.T,4063.T,8035.T,6857.T,6920.T,6367.T,4502.T,4503.T,6501.T,8001.T,8031.T,8058.T,9432.T,9433.T,2914.T,3382.T,4519.T,4568.T,4661.T,4901.T,6098.T,6146.T,6273.T,6503.T,6506.T,6594.T,6701.T,6702.T,6723.T,6902.T,6954.T,6981.T,7011.T,7267.T,7309.T,7733.T,7741.T,7832.T,8002.T,8113.T,8267.T,8316.T,8411.T,8591.T,8604.T,8630.T,8725.T,8766.T,8801.T,8802.T,9020.T,9021.T,9022.T,9101.T,9104.T,9107.T,9201.T,9202.T,9501.T,9502.T,9503.T,9613.T,9735.T,9843.T,9983.T",
        "💻 半導体・AI関連 (日米)": "NVDA,ASML,TSM,AMD,ARM,8035.T,6857.T,6146.T,PLTR,SMCI,AVGO,6723.T"
    }
    
    preset_choice = st.selectbox("カテゴリーから一括選択", options=list(PRESETS.keys()), key="preset_select_mass")
    ticker_list_raw = st.text_area(
        "対象ティッカー", 
        value=PRESETS[preset_choice] if PRESETS[preset_choice] else "AAPL,MSFT,NVDA...",
        key="ticker_input_mass"
    )
    
    use_parallel = st.checkbox("高速並列スキャンを使用 (推奨)", value=True)
    
    if st.button("🚀 大量スキャニング開始"):
        tickers = [t.strip() for t in ticker_list_raw.split(",") if t.strip()]
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_ticker(t_input):
            try:
                # 検索と整形
                target_t = t_input
                if target_t.isdigit() and len(target_t) == 4: target_t += ".T"
                
                # 1段階目: データ取得
                data, currency = get_stock_data(target_t, period=period)
                data = prepare_features(data)
                if len(data) < 50: return None
                
                # 2段階目: AI予測
                strategy = AIStrategy(api_key=api_key)
                strategy.train(data)
                signals = strategy.predict_signals(data)
                
                avg_pred = signals['Prediction'].tail(5).mean()
                annualized_pred = avg_pred * 252
                
                return {
                    "Ticker": target_t,
                    "Price": signals['Close'].iloc[-1],
                    "Currency": currency,
                    "AI Prediction": annualized_pred,
                    "R2": strategy.train(data)
                }
            except:
                return None

        # 並列実行
        if use_parallel:
            max_workers = 10 # 10スレッドで並列実行
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
                for i, future in enumerate(as_completed(future_to_ticker)):
                    res = future.result()
                    if res: results.append(res)
                    progress_bar.progress((i + 1) / len(tickers))
                    status_text.text(f"Processed {i+1}/{len(tickers)} stocks...")
        else:
            # 逐次実行（フォールバック）
            for i, t in enumerate(tickers):
                res = process_ticker(t)
                if res: results.append(res)
                progress_bar.progress((i + 1) / len(tickers))

        status_text.text("✅ 大量スキャン完了")
        
        if results:
            df_res = pd.DataFrame(results)
            # 年率リターンでソート
            df_res = df_res.sort_values(by="AI Prediction", ascending=False)
            
            # 通貨記号の付与
            df_res['Price_Display'] = df_res.apply(lambda x: f"{'¥' if x['Currency']=='JPY' else '$'}{x['Price']:,.1f}", axis=1)
            
            st.write(f"### 🏆 AI予測収益率ランキング (上位 {len(df_res)}件)")
            st.table(df_res[["Ticker", "Price_Display", "AI Prediction"]].head(50).style.format({"AI Prediction": "{:.2%}"}))
            
            # ポートフォリオへの追加を提案
            st.info("💡 上位銘柄を 'My Portfolio' に追加して、強化学習（RL）による詳細な売り時解析を行うことをお勧めします。")
                
            # 上位3件に対して詳細な投資戦略を生成
            st.subheader("💡 AI Recommended Investment Strategy (Top 3)")
            # 解析用のエンジンを定義
            report_strategy = AIStrategy(api_key=api_key)
            
            for _, row in df_res.head(3).iterrows():
                with st.expander(f"💰 {row['Ticker']} の投資戦略レポート"):
                    with st.spinner("Generating strategy..."):
                        try:
                            # ニュースを取得してセンチメントを計算
                            stock = yf.Ticker(row['Ticker'])
                            news = stock.news
                            s_score = 0
                            if news:
                                news_text = "\n".join([n.get('content', {}).get('title', '') for n in news[:5]])
                                try:
                                    import json
                                    raw_s = report_strategy.get_sentiment(row['Ticker'], news_text)
                                    clean_json = raw_s.strip()
                                    if "```json" in clean_json:
                                        clean_json = clean_json.split("```json")[1].split("```")[0].strip()
                                    res = json.loads(clean_json)
                                    s_score = res.get("score", 0)
                                except: pass
                            
                            # 投資アドバイスを生成
                            c_symbol = "¥" if row['Currency']=='JPY' else "$"
                            price_val = row['Price']
                            
                            advice = report_strategy.get_investment_advice(
                                row['Ticker'], price_val, row['AI Prediction'], 
                                s_score, total_budget, c_symbol
                            )
                            st.markdown(advice)
                        except Exception as e:
                            st.warning(f"レポート生成に失敗しました: {e}")
            else:
                st.info(f"期待収益率 {target_return_filter}% を超える銘柄は見つかりませんでした。")

with tab3:
    st.subheader("💰 My Portfolio Management")
    st.markdown("あなたの保有銘柄を登録し、AIからの売買アドバイスを確認します。")
    
    portfolio_df = load_portfolio()
    
    # 新規登録
    with st.expander("➕ 新しい持ち株を登録する"):
        st.info("💡 日本株は銘柄コード（例: 7203）を入力すると円建て、米国株（例: AAPL）はドル建てとして扱われます。")
        with st.form("add_holding_form"):
            new_ticker = st.text_input("銘柄コード (例: 7203, NVDA)")
            new_price = st.number_input("取得単価 (日本株は円、米国株はドル)", min_value=0.0, step=0.1)
            new_qty = st.number_input("数量", min_value=0, step=1)
            if st.form_submit_button("登録"):
                new_row = pd.DataFrame([{"Ticker": new_ticker, "Buy Price": new_price, "Quantity": new_qty}])
                portfolio_df = pd.concat([portfolio_df, new_row], ignore_index=True)
                save_portfolio(portfolio_df)
                st.success(f"{new_ticker} を登録しました！")
                st.rerun()

    if not portfolio_df.empty:
        st.subheader("現在の保有状況とAIアドバイス")
        
        # 表示用の空エリアを作成
        table_placeholder = st.empty()
        
        # 保有銘柄ごとにAI判断を実行
        portfolio_results = []
        for index, row in portfolio_df.iterrows():
            try:
                data, currency = get_stock_data(row['Ticker'], period="2y")
                data = prepare_features(data)
                
                strategy = AIStrategy(api_key=api_key)
                score = strategy.train(data)
                signals = strategy.predict_signals(data)
                
                current_price = signals['Close'].iloc[-1]
                prediction = signals['Prediction'].iloc[-1]
                
                advice, color_type = strategy.get_advice(current_price, row['Buy Price'], prediction)
                profit_pct = (current_price / row['Buy Price'] - 1) * 100
                
                # 通貨換算ロジック
                # 元の通貨
                native_currency = "JPY" if currency == "JPY" else "USD"
                
                # 表示用に変換
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
                    "Ticker": row['Ticker'],
                    "Qty": row['Quantity'],
                    f"Buy Price ({display_currency})": f"{currency_symbol}{disp_buy:,.1f}",
                    f"Current ({display_currency})": f"{currency_symbol}{disp_curr:,.1f}",
                    "Profit/Loss": f"{profit_pct:+.2f}%",
                    "AI Advice": advice,
                    "RL Strategy": "未実行 (下のボタンをクリック)"
                })
                
                # 表示の更新
                table_placeholder.table(pd.DataFrame(portfolio_results))
                
                # 詳細解析ボタン
                if st.button(f"🔍 {row['Ticker']} をRL詳細解析", key=f"rl_btn_{index}"):
                    with st.spinner(f"Analyzing {row['Ticker']} with RL..."):
                        strategy.train_rl(data, total_timesteps=500)
                        rl_advice_res, _ = strategy.get_rl_advice(data, row['Buy Price'])
                        st.success(f"**{row['Ticker']} RL判断:** {rl_advice_res}")
                
            except Exception as e:
                st.error(f"{row['Ticker']} の分析に失敗しました: {e}")
            
            # 個別削除機能
            st.markdown("### 🛠 銘柄の管理")
            cols = st.columns(len(portfolio_df))
            for i, (index, row) in enumerate(portfolio_df.iterrows()):
                if st.button(f"🗑 {row['Ticker']} を削除", key=f"del_{index}"):
                    portfolio_df = portfolio_df.drop(index)
                    save_portfolio(portfolio_df)
                    st.success(f"{row['Ticker']} を削除しました。")
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
