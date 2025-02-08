import sqlite3
import pandas as pd
import numpy as np
import logging
import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier  # 例としてRandomForestを追加
from sklearn.linear_model import LogisticRegression # 例としてロジスティック回帰を追加
from sklearn.svm import SVC # 例としてサポートベクターマシンを追加

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# データベース接続
DB_PATH = "stocks.db"

def fetch_data():
    logging.info("データ取得を開始")
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT code, datetime, open, high, low, close, volume FROM stocks"
        df = pd.read_sql(query, conn)
        conn.close()
        logging.info(f"データ取得完了: {df.shape}")
        return df
    except sqlite3.Error as e:
        logging.error(f"データベースエラー: {e}")
        return None

# RSI計算
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-6)  # ゼロ除算を避けるための微小値
    return 100 - (100 / (1 + rs))

# 特徴量作成
def create_features(df):
    logging.info("特徴量計算を開始")

    df = df.sort_values(by=["code", "datetime"])

    # 5日後の終値変化
    df["future_close"] = df.groupby("code")["close"].shift(-5)
    df["target"] = (df["future_close"] > df["close"]).astype(int)  # 1: 上昇, 0: 下落

    # 移動平均
    df["ma5"] = df.groupby("code")["close"].transform(lambda x: x.rolling(5).mean())
    df["ma10"] = df.groupby("code")["close"].transform(lambda x: x.rolling(10).mean())

    # ボリンジャーバンド
    df["bollinger_width"] = (df["high"] - df["low"]) / (df["close"] + 1e-7) # ゼロ除算対策

    # RSI
    df["rsi"] = df.groupby("code")["close"].transform(lambda x: compute_rsi(x))

    # 出来高変化率
    df["volume_change"] = df.groupby("code")["volume"].pct_change(5)
    df["volume_change"] = df["volume_change"].replace([np.inf, -np.inf], 0)  # 無限大を0で置き換え

    # 欠損値処理
    df.dropna(inplace=True)

    # 不要な列を削除
    df = df.drop(columns=["future_close"])

    logging.info(f"特徴量計算完了: {df.shape}")
    return df

# モデル学習
def train_model(X_train, y_train):
    logging.info("モデルの訓練を開始")

    # モデルのリスト
    models = {
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'), # solverを指定
        "SVM": SVC(random_state=42, probability=True) # probability=Trueを追加
    }

    best_models = {}

    for name, model in models.items():
        logging.info(f"{name} の学習を開始")
        param_grid = {}

        if name == "XGBoost":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 4, 5]
            }
        elif name == "RandomForest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5]
            }
        elif name == "LogisticRegression":
            param_grid = {
                "C": [0.1, 1, 10],  # 正則化パラメータ
                "penalty": ['l1', 'l2'] # 正則化の種類
            }
        elif name == "SVM":
            param_grid = {
                "C": [0.1, 1, 10],  # 正則化パラメータ
                "kernel": ['rbf', 'linear'] # カーネルの種類
            }

        grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        logging.info(f"最適な{name}ハイパーパラメータ: {grid_search.best_params_}")

    return best_models # 複数のモデルを返す

# モデル評価
def evaluate_model(model, X_test, y_test, model_name):
    logging.info(f"{model_name} の評価を開始")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # AUC-ROC計算のため確率を取得

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    logging.info(f"{model_name} 評価完了: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC-ROC={auc_roc:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    return acc, precision, recall, f1, auc_roc

# メイン処理
def main():
    start_time = time.time()

    logging.info("スクリプト開始")

    # データ取得
    df = fetch_data()
    if df is None:  # データ取得に失敗した場合
        return

    # 特徴量エンジニアリング
    df = create_features(df)

    # データ準備
    feature_columns = ["close", "ma5", "ma10", "bollinger_width", "rsi", "volume_change"]
    X = df[feature_columns]

    # 無限大やNaNが含まれていないか確認
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        logging.error("特徴量 X に無限大またはNaNが含まれています。")
        print(X) # 問題のある箇所を表示
        return  # 処理を中断

    y = df["target"]

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # クラスバランス調整（SMOTE適用）
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # モデル学習
    best_models = train_model(X_train_resampled, y_train_resampled) # 複数のモデルを取得

    # モデル評価
    for name, model in best_models.items():
        evaluate_model(model, X_test, y_test, name)

    # 結果をCSV保存 (ここではXGBoostの結果を例とする)
    df["predicted_xgb"] = best_models["XGBoost"].predict(X_scaled)
    filtered_stocks_xgb = df[df["predicted_xgb"] == 1][["code", "close", "rsi", "volume_change"]]
    filtered_stocks_xgb.to_csv("filtered_stocks_xgb.csv", index=False)
    logging.info("XGBoostによる銘柄リストを filtered_stocks_xgb.csv に保存しました。")

        # 結果をCSV保存 (ここではRandomForestの結果を例とする)
    df["predicted_rf"] = best_models["RandomForest"].predict(X_scaled)
    filtered_stocks_rf = df[df["predicted_rf"] == 1][["code", "close", "rsi", "volume_change"]]
    filtered_stocks_rf.to_csv("filtered_stocks_rf.csv", index=False)
    logging.info("RandomForestによる銘柄リストを filtered_stocks_rf.csv に保存しました。")

        # 結果をCSV保存 (ここではLogisticRegressionの結果を例とする)
    df["predicted_lr"] = best_models["LogisticRegression"].predict(X_scaled)
    filtered_stocks_lr = df[df["predicted_lr"] == 1][["code", "close", "rsi", "volume_change"]]
    filtered_stocks_lr.to_csv("filtered_stocks_lr.csv", index=False)
    logging.info("LogisticRegressionによる銘柄リストを filtered_stocks_lr.csv に保存しました。")

        # 結果をCSV保存 (ここではSVMの結果を例とする)
    df["predicted_svm"] = best_models["SVM"].predict(X_scaled)
    filtered_stocks_svm = df[df["predicted_svm"] == 1][["code", "close", "rsi", "volume_change"]]
    filtered_stocks_svm.to_csv("filtered_stocks_svm.csv", index=False)
    logging.info("SVMによる銘柄リストを filtered_stocks_svm.csv に保存しました。")

    end_time = time.time()
    logging.info(f"スクリプト終了（実行時間: {end_time - start_time:.2f} 秒）")

if __name__ == "__main__":
    main()
