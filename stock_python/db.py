# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 22:58:06 2022

@author: dearl
"""

import openpyxl
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import date, datetime
import pandas as pd
import sqlite3

def get_codes(file_path = r"/python/東京株式市場.xlsx"):    
    skip_words =['ETF・ETN', 'PRO Market']
    codes = []

    wb = openpyxl.load_workbook(file_path)
    ws = wb["Sheet1"]
    for row in ws.iter_rows(min_row=2):
        market = str(row[3].value)
        if (not exist_words(market, skip_words)):
            codes.append(str(row[1].value))
    return codes

def exist_words(text, words):
    exist = False
    for word in words:
        if (word in text):
            exist = True
    return exist


# 東証コードのリストを取得
codes = get_codes(file_path = r"/python/東京株式市場.xlsx") 

# SQLite3に接続
conn = sqlite3.connect("stocks.db")
c = conn.cursor()

# テーブルを作成
c.execute('''CREATE TABLE IF NOT EXISTS stocks (code TEXT, datetime TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)''')
conn.commit()

for code in codes:
    print(code)
    my_share = share.Share(code + ".T")
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_YEAR,
                                            3,
                                            share.FREQUENCY_TYPE_DAY,
                                            1)
    except YahooFinanceError as e:
        print(e.message)

    print(symbol_data)
    
    # csv形式で保存
    if (symbol_data == None):
        continue

    df = pd.DataFrame({'datetime': [datetime.fromtimestamp(timestamp / 1000) for timestamp in symbol_data['timestamp']],\
        'open' : symbol_data['open'], 'high' : symbol_data['high'],
        'low' : symbol_data['low'], 'close' : symbol_data['close'], 'volume' : symbol_data['volume']})

    # SQLite3にインサート
    for index, row in df.iterrows():
        d = datetime.now()

        c.execute('''INSERT INTO stocks (code, datetime, open, high, low, close, volume) 
                     VALUES ('{0}', ?, ?, ?, ?, ?, ?)'''.format(code), 
                  (row['datetime'], row['open'], row['high'], row['low'], row['close'], row['volume']))

    conn.commit()

# SQLite3との接続を閉じる
conn.close()
