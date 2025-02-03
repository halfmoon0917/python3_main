import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
import logging
import time

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_database(db_path, retries=5, delay=2):
    for attempt in range(retries):
        try:
            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            logging.info("データベースに接続しました。")
            return conn
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < retries - 1:
                logging.warning(f"データベースがロックされています。{delay}秒後に再試行します... ({attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                logging.error(f"データベース接続エラー: {e}")
                raise

def fetch_data_in_chunks(conn, query, chunksize=5000):
    try:
        chunks = []
        for chunk in pd.read_sql(query, conn, chunksize=chunksize):
            chunks.append(chunk)
        result = pd.concat(chunks, ignore_index=True)
        logging.info(f"データを取得しました: {len(result)} 行")
        return result
    except Exception as e:
        logging.error(f"データ取得エラー: {e}")
        raise

def calculate_features(df):
    try:
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma25'] = df['close'].rolling(window=25, min_periods=1).mean()
        df['return'] = df['close'].pct_change().fillna(0)
        df['volatility'] = df['return'].rolling(window=5, min_periods=1).std().fillna(0)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['volume_change'] = df['volume'].pct_change().fillna(0)
        df['bollinger_upper'] = df['ma25'] + 2 * df['close'].rolling(window=25, min_periods=1).std()
        df['bollinger_lower'] = df['ma25'] - 2 * df['close'].rolling(window=25, min_periods=1).std()
        df.bfill(inplace=True)
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
    except Exception as e:
        logging.error(f"特徴量計算エラー: {e}")
        raise
    return df

def train_model_and_predict(df, features, output_csv="filtered_stocks.csv", min_volume=100000):
    df['price_increase_5days'] = (df['close'].shift(-5) >= df['close'] * 1.05).astype(int)
    df['predicted_close_5days'] = df['close'].shift(-5)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df[features]
    y = df['price_increase_5days']
    X_scaled = StandardScaler().fit_transform(X)
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, n_estimators=500, max_depth=20, min_samples_split=5, min_samples_leaf=1, max_features='sqrt')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, y_scores)
    else:
        auc = float('nan')
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"モデル評価 - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    df['probability'] = model.predict_proba(X_scaled)[:, 1]
    df['predicted'] = (df['probability'] >= 0.7).astype(int)
    df['datetime'] = pd.to_datetime(df['datetime'])
    latest_date = df['datetime'].max()
    df = df[df['datetime'] == latest_date]
    df = df[df['volume'] >= min_volume]
    selected_stocks = df.sort_values(by='probability', ascending=False).head(200)[['code', 'datetime', 'close', 'predicted_close_5days', 'probability']]
    selected_stocks.to_csv(output_csv, index=False)
    logging.info(f"スイングトレード向けの銘柄を {output_csv} に保存しました。")
    return model

if __name__ == "__main__":
    db_path = "stocks.db"
    query = "SELECT * FROM stocks"
    conn = connect_database(db_path)
    df = fetch_data_in_chunks(conn, query)
    df = calculate_features(df)
    features = ['ma5', 'ma25', 'return', 'volatility', 'rsi', 'volume_change', 'bollinger_upper', 'bollinger_lower', 'is_bullish']
    model = train_model_and_predict(df, features, output_csv="final_filtered_stocks.csv")
    conn.close()
