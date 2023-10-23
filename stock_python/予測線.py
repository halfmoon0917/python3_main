import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# SQLiteデータベースに接続
conn = sqlite3.connect("stocks.db")

# 特徴量として使用するデータを取得
code = '7211'  # 予測対象の株価コード
query = f"SELECT datetime, close FROM stocks WHERE code='{code}' ORDER BY datetime ASC"
df = pd.read_sql_query(query, conn)

# 予測対象の日数
prediction_days = 30

# 特徴量エンジニアリング：過去の株価を特徴量として使用
for i in range(1, prediction_days + 1):
    df[f'close-{i}'] = df['close'].shift(i)

df['datetime'] = pd.to_datetime(df['datetime'])  # 'datetime' 列を日付時刻の型に変換

# 最後のN日を予測対象として取得
X = df.dropna().drop(['datetime', 'close'], axis=1)
y = df['close'].shift(-prediction_days).dropna()

# データをトレーニングセットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ランダムフォレスト回帰モデルの作成
model = RandomForestRegressor(n_estimators=100, random_state=42)

# モデルのトレーニング
model.fit(X_train, y_train)

# 予測の実行
y_pred = model.predict(X_test)

# モデルの評価：平均二乗誤差を計算
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# 未来の株価を予測するためのデータを作成
last_date = pd.to_datetime(df['datetime'].iloc[-1])  # 最後の日付を取得し、日付型に変換
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, prediction_days + 1)]
future_price = model.predict(X.iloc[-prediction_days:, :])

# 予測結果をデータフレームに追加
future_df = pd.DataFrame({'datetime': future_dates, 'close': future_price})
df = df.append(future_df, ignore_index=True)

conn.commit()

# 予測と実際の株価をプロット
plt.figure(figsize=(18, 6))  # グラフの幅を大きくする
plt.plot(range(len(y_test)), y_test.values, label="real_price", color='blue')
plt.xticks(range(len(y_test)), df['datetime'][-len(y_test):].dt.date, rotation=45)  # 日付のみ表示
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()



# データベース接続を閉じる
conn.close()
