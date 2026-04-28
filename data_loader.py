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
    df = t.history(period=period, interval=interval)
    
    if isinstance(df, dict) or df is None or (hasattr(df, 'empty') and df.empty):
        raise ValueError(f"Ticker '{ticker}' のデータが見つかりませんでした。")
        
    # yahooqueryの出力形式を調整 (マルチインデックスの解除とカラム名の正規化)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index().set_index('date')
    elif 'date' in df.columns:
        df = df.set_index('date')
        
    # カラム名を大文字開始に統一 (yfinance 互換)
    rename_map = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume', 'adjclose': 'Adj Close'
    }
    df = df.rename(columns=rename_map)
    
    # 型変換とインデックス調整
    df.index = pd.to_datetime(df.index).tz_localize(None)
    
    # 通貨情報の取得 (yahooquery)
    try:
        summary = t.summary_detail.get(ticker, {})
        currency = summary.get('currency', 'USD')
    except:
        currency = 'USD'
        
    return df, currency

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
