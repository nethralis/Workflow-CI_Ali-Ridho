import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Aktifkan Autolog
# Autolog ini pintar, dia bakal otomatis nempel ke Run ID yang dibuat oleh 'mlflow run'
mlflow.autolog()

def main(train_data, test_data, target_column):
    # Baca data dari parameter input
    print(f"Loading data from: {train_data}")
    df = pd.read_csv(train_data)

    # Pisahkan Fitur dan Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- BAGIAN YANG DIPERBAIKI (HAPUS start_run) ---
    # Kita langsung training aja. Karena script ini dijalankan via 'mlflow run',
    # dia otomatis sudah punya Active Run. Gak perlu bikin baru.
    
    print("Starting training...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc}")
    
    # KITA TIDAK PERLU mlflow.log_metric manual karena sudah ada mlflow.autolog()
    # Tapi kalau mau nambah log manual, langsung aja panggil (gak usah pake with start_run)
    mlflow.log_metric("manual_accuracy", acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--target_column", type=str)
    args = parser.parse_args()

    main(args.train_data, args.test_data, args.target_column)
