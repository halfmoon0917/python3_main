import sqlite3
import pandas as pd
import talib
from datetime import datetime

# Create a connection to the SQLite database
conn = sqlite3.connect("stocks.db")

# Set the initial parameters
lookback = 100
rsi_value = 30

# Define the criteria for a buy signal
def is_buy_signal(df):
    # Calculate the indicators
    ma5 = talib.SMA(df['close'], timeperiod=5)
    ma25 = talib.SMA(df['close'], timeperiod=25)
    rsi = talib.RSI(df['close'], timeperiod=14)
    
    # Check the criteria for a buy signal
    if (ma25[-3] < ma25[-2] < ma25[-1] and
        ma5[-3] < ma5[-2] < ma5[-1] and
        rsi[-1] < rsi_value):
        return True
    else:
        return False

# Get the list of codes
c = conn.cursor()
c.execute("SELECT DISTINCT code FROM stocks")
codes = [row[0] for row in c.fetchall()]

buy_signals = []
# Search for a buy signal, gradually relaxing the criteria if no signals are found
for i in range(5):
    for code in codes:
        # Get the historical prices for the given code
        query = f"SELECT code, datetime, close FROM stocks WHERE code='{code}' ORDER BY datetime DESC LIMIT {lookback}"
        df = pd.read_sql_query(query, conn)
        df = df.set_index('datetime')
        df['close'] = df['close'].astype(float)  # Convert 'close' column to float
        df.sort_index(ascending=True, inplace=True)  # Sort the dataframe by datetime in ascending order
        df.index = pd.to_datetime(df.index)  # Convert index to datetime format
        
        # Check if the data meets the lookback period requirement
        if len(df) < lookback:
            continue
        
        # Check for a buy signal
        if is_buy_signal(df):
            signal_date = df.index[-1].strftime('%Y-%m-%d')
            last_price = df['close'].iloc[-1]
            buy_signals.append({"code": code, "date": signal_date, "price": last_price})

    # If buy signals were found, print them and export to CSV
    if len(buy_signals) > 0:
        sorted_signals = sorted(buy_signals, key=lambda x: x['code'])
        print(f"Buy signals found with the specified conditions (Attempt {i+1}):")
        for signal in sorted_signals:
            print(f"Code: {signal['code']}, Price: {signal['price']}, Date: {signal['date']}")
        
        # Export to CSV
        filename = f"buy_signals_{lookback}_days_RSI_lt_{rsi_value}.csv"
        df = pd.DataFrame(sorted_signals)
        df.to_csv(filename, index=False, columns=['code', 'price', 'date'])
        break

    # If no buy signals were found, gradually relax the criteria and try again
    lookback += 50
    rsi_value += 5
    print(f"No buy signals found with {lookback} days lookback and RSI < {rsi_value}. Relaxing criteria...")
