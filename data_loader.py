import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period: str = "2y", interval: str = "1d"):
    """
    yfinanceを使用して株価データを取得する。
    """
    # 正規化
    ticker = ticker.strip().upper()
    if ticker.isdigit() and len(ticker) == 4:
        ticker += ".T"
        
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"Ticker '{ticker}' のデータが見つかりませんでした。")
        
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df, stock.info.get('currency', 'USD')

def search_ticker(query: str):
    """
    キーワードからティッカーシンボルを検索する。
    """
    try:
        print(f"Searching for: {query}")
        # yfinanceのSearch機能を使用
        # 日本語の場合は、内部的に一度検索を試みる
        search = yf.Search(query, max_results=10)
        results = search.quotes
        
        if not results:
            return None
            
        # 候補の中から最適なものを選択
        # 1. 日本株 (.T) を最優先
        for res in results:
            symbol = res.get('symbol', '')
            if symbol.endswith('.T'):
                return symbol
        
        # 2. 米国株（アルファベットのみのシンボル）を次点
        for res in results:
            symbol = res.get('symbol', '')
            if symbol.isalpha():
                return symbol
                
        # 3. それ以外（指数など）
        return results[0].get('symbol')
        
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
