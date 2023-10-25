from datetime import datetime
import pandas as pd
import sqlite3
import talib

# SQLiteデータベースに接続
conn = sqlite3.connect("stocks.db")

# Set the initial parameters
lookback_weekly = 75  # 週足のデータ取得期間

# Define the criteria for a long entry signal based on weekly moving averages (上昇トレンド)
def is_long_entry_signal(df):
    # Calculate the weekly moving averages (75週線、25週線、5週線)
    ma5 = talib.SMA(df['close'], timeperiod=5)
    ma25 = talib.SMA(df['close'], timeperiod=25)
    ma75 = talib.SMA(df['close'], timeperiod=75)
    
    # Check the criteria for a long entry signal (上昇トレンド)
    if (ma75[-3] < ma25[-3] < ma5[-3] and
        ma75[-2] < ma25[-2] < ma5[-2] and
        ma75[-1] < ma25[-1] < ma5[-1]):
        return True
    else:
        return False

# Define the criteria for a short entry signal based on weekly moving averages (下降トレンド)
def is_short_entry_signal(df):
    # Calculate the weekly moving averages (75週線、25週線、5週線)
    ma5 = talib.SMA(df['close'], timeperiod=5)
    ma25 = talib.SMA(df['close'], timeperiod=25)
    ma75 = talib.SMA(df['close'], timeperiod=75)
    
    # Check the criteria for a short entry signal (下降トレンド)
    if (ma75[-3] > ma25[-3] > ma5[-3] and
        ma75[-2] > ma25[-2] > ma5[-2] and
        ma75[-1] > ma25[-1] > ma5[-1]):
        return True
    else:
        return False

# Get the list of codes
c = conn.cursor()
c.execute("SELECT DISTINCT code FROM stocks")
codes = [row[0] for row in c.fetchall()]

long_entry_signals = []
short_entry_signals = []

# Search for entry signals, gradually relaxing the criteria if no signals are found
for code in codes:
    # Get the historical prices for the given code (日足)
    query_daily = f"SELECT code, datetime, close FROM stocks WHERE code='{code}'"
    df_daily = pd.read_sql_query(query_daily, conn)
    df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
    df_daily.set_index('datetime', inplace=True)
    
    # Resample daily data to weekly OHLCV data
    df_weekly = df_daily['close'].resample('W').agg({
        'code': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Check if the data meets the lookback period requirement for weekly data
    if len(df_weekly) < lookback_weekly:
        continue

    # ...

    # 週足データを使用して条件を判定
    if is_long_entry_signal(df_weekly):
        signal_date = df_weekly.index[-1].strftime('%Y-%m-%d')
        last_price = df_weekly['close'].iloc[-1]
        long_entry_signals.append({"code": code, "date": signal_date, "price": last_price})
    
    if is_short_entry_signal(df_weekly):
        signal_date = df_weekly.index[-1].strftime('%Y-%m-%d')
        last_price = df_weekly['close'].iloc[-1]
        short_entry_signals.append({"code": code, "date": signal_date, "price": last_price})

# Output long entry signals to CSV
if len(long_entry_signals) > 0:
    sorted_long_signals = sorted(long_entry_signals, key=lambda x: x['code'])
    print("Long entry signals found with the specified conditions:")
    for signal in sorted_long_signals:
        print(f"Code: {signal['code']}, Price: {signal['price']}, Date: {signal['date']}")
    
    filename_long = f"long_entry_signals_WeeklyMovingAverages.csv"
    df_long = pd.DataFrame(sorted_long_signals)
    df_long.to_csv(filename_long, index=False, columns=['code', 'price', 'date'])
else:
    print("No long entry signals found with the specified conditions.")

# Output short entry signals to CSV
if len(short_entry_signals) > 0:
    sorted_short_signals = sorted(short_entry_signals, key=lambda x: x['code'])
    print("Short entry signals found with the specified conditions:")
    for signal in sorted_short_signals:
        print(f"Code: {signal['code']}, Price: {signal['price']}, Date: {signal['date']}")
    
    filename_short = f"short_entry_signals_WeeklyMovingAverages.csv"
    df_short = pd.DataFrame(sorted_short_signals)
    df_short.to_csv(filename_short, index=False, columns=['code', 'price', 'date'])
else:
    print("No short entry signals found with the specified conditions.")


# JPX 400の銘柄リストを読み込む
jpx_400_df = pd.read_csv("jpx_400.csv")

# Long entry signalsのデータを読み込む
long_entry_df = pd.read_csv("long_entry_signals_WeeklyMovingAverages.csv")

# JPX 400に含まれる銘柄のみを抽出
filtered_long_entry_df = long_entry_df[long_entry_df['code'].isin(jpx_400_df['コード'])]

# 結果をCSVファイルに保存
filtered_long_entry_df.to_csv("long_400.csv", index=False)

# Short entry signalsのデータを読み込む
short_entry_df = pd.read_csv("short_entry_signals_WeeklyMovingAverages.csv")

# JPX 400に含まれる銘柄のみを抽出
filtered_short_entry_df = short_entry_df[short_entry_df['code'].isin(jpx_400_df['コード'])]

# 結果をCSVファイルに保存
filtered_short_entry_df.to_csv("short_400.csv", index=False)

# Long entry signalsのデータをpriceが1000.0以下の行のみでフィルタリング
filtered_long_entry_df = filtered_long_entry_df[filtered_long_entry_df['price'] <= 1000.0]

# 結果をCSVファイルに保存
filtered_long_entry_df.to_csv("long_400_filtered.csv", index=False)

# Short entry signalsのデータをpriceが1000.0以下の行のみでフィルタリング
filtered_short_entry_df = filtered_short_entry_df[filtered_short_entry_df['price'] <= 1000.0]

# 結果をCSVファイルに保存
filtered_short_entry_df.to_csv("short_400_filtered.csv", index=False)

