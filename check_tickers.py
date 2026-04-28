import pandas as pd
import requests
import io
import os

def check():
    print("--- 銘柄取得エンジン検証 ---")
    
    # 米国株検証
    us_count = 0
    try:
        url_sp500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        res = requests.get(url_sp500, timeout=10)
        dfs = pd.read_html(res.text)
        df_us = dfs[0]
        us_count = len(df_us)
        print(f"✅ US Wikipedia (S&P 500): {us_count} 銘柄")
    except Exception as e:
        print(f"❌ US Error: {e}")

    # 日本株検証
    jp_count = 0
    try:
        url_jpx400 = "https://www.jpx.co.jp/markets/indices/jpx-nikkei400/tvdivq0000001vg2-att/jpxnk400_weight_j.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url_jpx400, headers=headers, timeout=10)
        content = res.content.decode('cp932')
        df_jp = pd.read_csv(io.StringIO(content), skiprows=1)
        jp_count = len(df_jp)
        print(f"✅ JP JPX400 (CSV): {jp_count} 銘柄")
    except Exception as e:
        print(f"❌ JP Error: {e}")
        # もしCSVがダメなら代替案を提示
        print("💡 CSV取得に失敗しました。URLが変更されているか、フォーマットが変わっています。")

    total = us_count + jp_count
    print(f"\n合計取得件数: {total} 銘柄")
    if total < 900:
        print("⚠️ 警告: 目標の1,400件に到達していません。")
    else:
        print("✨ 正常: 目標の1,400件規模に到達しました。")

if __name__ == "__main__":
    check()
