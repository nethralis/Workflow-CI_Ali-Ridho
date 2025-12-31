import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Aktifkan Autolog
mlflow.autolog()

def main(train_data, test_data, target_column):
    # Baca data dari parameter input
    print(f"Loading data from: {train_data}")
    
    # --- UPDATE BIAR AMAN ---
    # Gunakan sep=None dan engine='python' biar dia deteksi otomatis (koma atau titik koma)
    try:
        df = pd.read_csv(train_data, sep=None, engine='python')
    except:
        # Fallback kalau error, coba baca standar
        df = pd.read_csv(train_data)

    # --- DEBUGGING (PENTING) ---
    # Ini bakal nge-print nama-nama kolom ke log GitHub biar kita tahu isinya apa
    print("DATA SHAPE:", df.shape)
    print("COLUMNS FOUND:", df.columns.tolist())
    
    # Cek apakah target ada?
    if target_column not in df.columns:
        # Coba bersihkan spasi di nama kolom (kadang ada spasi nyempil: " Exited")
        df.columns = df.columns.str.strip()
        print("Cleaned Columns:", df.columns.tolist())
        
        if target_column not in df.columns:
            raise KeyError(f"Kolom target '{target_column}' GAK KETEMU! Yang ada cuma: {df.columns.tolist()}")

    # Pisahkan Fitur dan Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training
    print("Starting training...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluasi
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--target_column", type=str)
    args = parser.parse_args()

    main(args.train_data, args.test_data, args.target_column)
