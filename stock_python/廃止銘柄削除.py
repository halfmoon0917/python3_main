import openpyxl
import sqlite3
import csv

def get_codes(file_path="東京株式市場.xlsx"):
    skip_words = ['ETF・ETN', 'PRO Market']
    codes = []

    # エクセルファイルの読み込み
    wb = openpyxl.load_workbook(file_path)
    ws = wb["Sheet1"]

    # コードを収集
    for row in ws.iter_rows(min_row=2):
        market = str(row[3].value)
        if not exist_words(market, skip_words):
            code = str(row[1].value)
            if code:  # コードがNoneまたは空でないことを確認
                codes.append(code)
    return codes

def exist_words(text, words):
    return any(word in text for word in words)

def export_unmatched_to_csv(db_path, unmatched_codes, output_csv="unmatched_codes.csv"):
    # SQLiteデータベースへの接続
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 削除対象のコードに関連するレコードをCSVに出力
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["code", "datetime", "open", "high", "low", "close", "volume"])  # ヘッダー

        for code in unmatched_codes:
            c.execute("SELECT * FROM stocks WHERE code = ?", (code,))
            rows = c.fetchall()
            writer.writerows(rows)  # レコードをCSVに書き込む

    conn.close()
    print(f"Unmatched records exported to {output_csv}")

def delete_unmatched_codes(db_path, unmatched_codes):
    # SQLiteデータベースへの接続
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # 存在しないコードを削除
    for code in unmatched_codes:
        c.execute("DELETE FROM stocks WHERE code = ?", (code,))
        print(f"Deleted records with code: {code}")

    # 変更を保存して接続を閉じる
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # エクセルからコードを取得
    excel_file_path = "東京株式市場.xlsx"
    excel_codes = get_codes(excel_file_path)

    # データベースからエクセルにないコードを特定
    db_path = "stocks.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT DISTINCT code FROM stocks")
    db_codes = {row[0] for row in c.fetchall()}  # データベースのコードをセットで取得
    conn.close()

    unmatched_codes = db_codes - set(excel_codes)

    # 削除対象のコードをCSVに出力
    if unmatched_codes:
        export_unmatched_to_csv(db_path, unmatched_codes)

        # 削除処理を実行
        delete_unmatched_codes(db_path, unmatched_codes)
    else:
        print("No unmatched codes found. No records to delete.")
