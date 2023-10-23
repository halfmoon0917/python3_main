import openpyxl
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import datetime
import pandas as pd
import sqlite3

def get_codes(file_path="東京株式市場.xlsx"):
    skip_words = ['ETF・ETN', 'PRO Market']
    codes = []

    wb = openpyxl.load_workbook(file_path)
    ws = wb["Sheet1"]
    for row in ws.iter_rows(min_row=2):
        market = str(row[3].value)
        if not exist_words(market, skip_words):
            codes.append(str(row[1].value))
    return codes


def exist_words(text, words):
    exist = False
    for word in words:
        if word in text:
            exist = True
    return exist


# Create a connection to the SQLite database
conn = sqlite3.connect("stocks.db")

# Create a cursor object to execute SQL commands
c = conn.cursor()

# Create the 'stocks' table if it doesn't already exist
c.execute('''CREATE TABLE IF NOT EXISTS stocks
             (code TEXT, datetime TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)''')

# Get the list of codes
codes = get_codes()

for code in codes:
    print(code)
    my_share = share.Share(code + ".T")
    symbol_data = None

    try:
        symbol_data = my_share.get_historical(share.PERIOD_TYPE_DAY,
                                            1,
                                            share.FREQUENCY_TYPE_DAY,
                                            1)
    except YahooFinanceError as e:
        print(e.message)

    print(symbol_data)

    # sqlite3 に保存
    if symbol_data is not None:
        data = [(code, datetime.fromtimestamp(d / 1000), o, h, l, c, v) for d, o, h, l, c, v in zip(symbol_data['timestamp'], symbol_data['open'], symbol_data['high'], symbol_data['low'], symbol_data['close'], symbol_data['volume'])]
        for row in data:
            c.execute('''SELECT * FROM stocks WHERE code = ? AND datetime = ?''', (row[0], row[1]))
            if c.fetchone():
                c.execute('''UPDATE stocks SET open=?, high=?, low=?, close=?, volume=? WHERE code = ? AND datetime = ?''', (row[2], row[3], row[4], row[5], row[6], row[0], row[1]))
            else:
                c.execute('''INSERT INTO stocks (code, datetime, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)''', row)
        conn.commit()

# Close the connection to the database
conn.close()
