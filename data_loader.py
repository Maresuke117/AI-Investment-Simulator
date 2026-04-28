from yahooquery import Ticker, search as yq_search
import pandas as pd
import numpy as np

def get_stock_data(ticker: str, period: str = "2y", interval: str = "1d"):
    """
    yfinanceを使用して株価データを取得する。
    """
    # 正規化
    ticker = ticker.strip().upper()
    if ticker.isdigit() and len(ticker) == 4:
        ticker += ".T"
        
    print(f"Fetching data for {ticker} using yahooquery...")
    t = Ticker(ticker)
    
    # リトライロジック (簡易)
    for _ in range(2):
        df = t.history(period=period, interval=interval)
        if df is not None and not (hasattr(df, 'empty') and df.empty):
            break
        time.sleep(0.5)
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        raise ValueError(f"Ticker '{ticker}' のデータが見つかりませんでした。")
        
    # エラーメッセージが返ってきた場合の処理
    if isinstance(df, dict):
        error_msg = df.get(ticker, "Unknown Error")
        raise ValueError(f"Ticker '{ticker}' でエラーが発生しました: {error_msg}")
        
    # yahooqueryの出力形式を調整
    try:
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'Date' in df.columns:
                df = df.set_index('Date')
        elif 'date' in df.columns:
            df = df.set_index('date')
        elif 'Date' in df.columns:
            df = df.set_index('Date')
    except Exception as e:
        print(f"Index adjustment failed for {ticker}: {e}")
        
    # カラム名を大文字開始に統一 (yfinance 互換)
    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume', 'adjclose': 'Adj Close'
    }
    df = df.rename(columns=rename_map)
    
    # 通貨情報の取得 (安全な取得)
    currency = "JPY" if ticker.endswith('.T') else "USD"
    name = ticker
    try:
        # 取得したDataFrameのインデックスを日付型に変換
        # MultiIndex(symbol, date) の場合は symbol を消して date だけにする
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)
        
        # インデックスを確実に日付型にしてタイムゾーンを消す
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # メタデータの取得 (直接アクセスを避けて get_modules を使用)
        modules = t.get_modules(['summaryDetail', 'price'])
        if isinstance(modules, dict) and ticker in modules:
            m_data = modules[ticker]
            if isinstance(m_data, dict):
                # price モジュール
                p_info = m_data.get('price', {})
                if isinstance(p_info, dict):
                    currency = p_info.get('currency') or currency
                    name = p_info.get('shortName') or name
                
                # summaryDetail モジュール
                s_detail = m_data.get('summaryDetail', {})
                if isinstance(s_detail, dict):
                    currency = s_detail.get('currency') or currency
    except Exception as e:
        print(f"Metadata/Index adjustment failed for {ticker}: {e}")
        
    return df, currency, name

def search_ticker(query: str):
    """
    キーワードからティッカーシンボルを検索する。
    """
    try:
        print(f"Searching for: {query}")
        results = yq_search(query)
        quotes = results.get('quotes', [])
        
        if not quotes:
            return None
            
        # 1. 日本株 (.T) を最優先
        for res in quotes:
            symbol = res.get('symbol', '')
            if symbol.endswith('.T'):
                return symbol
        
        # 2. 米国株
        for res in quotes:
            symbol = res.get('symbol', '')
            if symbol.isalpha() and not symbol.isdigit():
                return symbol
                
        return quotes[0].get('symbol')
        
    except Exception as e:
        print(f"Search error for '{query}': {e}")
    return None

if __name__ == "__main__":
    # テスト実行
    try:
        data = get_stock_data("AAPL")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
