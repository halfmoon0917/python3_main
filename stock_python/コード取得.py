import openpyxl
import yfinance as yf
from datetime import datetime, timedelta
import sqlite3
import logging

# ロギングの設定
logging.basicConfig(filename='stock_data_download.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s', filemode='w')

# コードのリストを取得する関数（例: Excelファイルから取得）
def exist_words(text, words):
    if text is None:
        return False
    return any(word in text for word in words)

def get_codes(file_path="東京株式市場.xlsx"):
    skip_words = ['ETF・ETN', 'PRO Market']
    codes = []
    
    try:
        wb = openpyxl.load_workbook(file_path, data_only=True)
        ws = wb["Sheet1"]
        
        for row in ws.iter_rows(min_row=2, values_only=True):
            code = row[1]  # 銘柄コード（2列目）
            market = row[3]  # 市場（4列目）
            
            if code and isinstance(code, (int, str)) and not exist_words(str(market), skip_words):
                codes.append(str(code))

        wb.close()
        
    except Exception as e:
        logging.error(f"Excel 読み込みエラー: {e}")
    
    return codes

# SQLiteデータベースへの接続
conn = sqlite3.connect("stocks.db")
c = conn.cursor()

# コードのリストを取得
codes = get_codes()

today = datetime.today().strftime('%Y-%m-%d')

def get_missing_dates(code):
    """ 過去5日間で欠損している日付を取得 """
    missing_dates = []
    
    for i in range(1, 6):  # 過去5日分
        check_date = (datetime.today() - timedelta(days=i)).strftime('%Y-%m-%d')
        c.execute('''SELECT COUNT(*) FROM stocks WHERE code = ? AND datetime = ?''', (code, check_date))
        if c.fetchone()[0] == 0:
            missing_dates.append(check_date)
    
    return missing_dates

# データ取得の処理
for code in codes:
    ticker = f"{code}.T"
    
    # DBから最新の日付を取得
    c.execute('''SELECT MAX(datetime) FROM stocks WHERE code = ?''', (code,))
    latest_date = c.fetchone()[0]
    
    # 通常の取得（最新の翌日～昨日）
    if latest_date:
        start_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    end_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    if start_date <= end_date:
        logging.info(f"データ取得開始: {ticker} の {start_date} から {end_date} までのデータを取得")
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
        if not df.empty:
            for dt, row in df.iterrows():
                data = (
                    code,
                    dt.strftime('%Y-%m-%d'),
                    float(row['Open'].iloc[0]) if not row['Open'].isna().all() else 0.0,
                    float(row['High'].iloc[0]) if not row['High'].isna().all() else 0.0,
                    float(row['Low'].iloc[0]) if not row['Low'].isna().all() else 0.0,
                    float(row['Close'].iloc[0]) if not row['Close'].isna().all() else 0.0,
                    int(row['Volume'].iloc[0]) if not row['Volume'].isna().all() else 0
                )
                
                # 既存データの確認
                c.execute('''SELECT * FROM stocks WHERE code = ? AND datetime = ?''', (data[0], data[1]))
                if c.fetchone():
                    c.execute('''UPDATE stocks 
                                 SET open=?, high=?, low=?, close=?, volume=? 
                                 WHERE code = ? AND datetime = ?''', 
                              (data[2], data[3], data[4], data[5], data[6], data[0], data[1]))
                else:
                    c.execute('''INSERT INTO stocks (code, datetime, open, high, low, close, volume) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?)''', data)
            conn.commit()

    # 欠損データの補完
    missing_dates = get_missing_dates(code)
    
    for missing_date in missing_dates:
        df = yf.download(ticker, start=missing_date, end=(datetime.strptime(missing_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'), interval="1d")
        
        if not df.empty:
            for dt, row in df.iterrows():
                data = (
                    code,
                    dt.strftime('%Y-%m-%d'),
                    float(row['Open'].iloc[0]) if not row['Open'].isna().all() else 0.0,
                    float(row['High'].iloc[0]) if not row['High'].isna().all() else 0.0,
                    float(row['Low'].iloc[0]) if not row['Low'].isna().all() else 0.0,
                    float(row['Close'].iloc[0]) if not row['Close'].isna().all() else 0.0,
                    int(row['Volume'].iloc[0]) if not row['Volume'].isna().all() else 0
                )
                
                c.execute('''SELECT * FROM stocks WHERE code = ? AND datetime = ?''', (data[0], data[1]))
                if c.fetchone():
                    c.execute('''UPDATE stocks 
                                 SET open=?, high=?, low=?, close=?, volume=? 
                                 WHERE code = ? AND datetime = ?''', 
                              (data[2], data[3], data[4], data[5], data[6], data[0], data[1]))
                else:
                    c.execute('''INSERT INTO stocks (code, datetime, open, high, low, close, volume) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?)''', data)
            conn.commit()

# 最新データの取得
c.execute('''SELECT * FROM stocks ORDER BY datetime DESC LIMIT 1''')
latest_record = c.fetchone()

if latest_record:
    print(f"\n最新のレコード: {latest_record}")
else:
    print("\nデータベースにはデータがありません")

# 直近5日間の日ごとのデータ件数を取得し出力
def get_recent_daily_counts():
    print("\n【直近5日間の日毎のデータ件数】")
    print("日付       | 件数")
    print("--------------------")
    
    five_days_ago = (datetime.today() - timedelta(days=5)).strftime('%Y-%m-%d')

    c.execute('''
        SELECT datetime, COUNT(*)
        FROM stocks
        WHERE datetime >= ?
        GROUP BY datetime
        ORDER BY datetime ASC
    ''', (five_days_ago,))
    
    results = c.fetchall()
    
    for date, count in results:
        print(f"{date} | {count}")

get_recent_daily_counts()

# 接続を閉じる
conn.close()
