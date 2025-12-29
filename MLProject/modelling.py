import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

# --- CONFIG ---
# Script akan otomatis cari file csv di folder yang sama
INPUT_FILE = "bank_data_preprocessing.csv"

if __name__ == "__main__":
    print(f"Mencari dataset: {INPUT_FILE} ...")
    
    # Cek apakah file ada (Anti-Error Path)
    if not os.path.exists(INPUT_FILE):
        print("ERROR FATAL: File csv tidak ditemukan!")
        print(f"Pastikan file '{INPUT_FILE}' ada di sebelah file modelling.py ini.")
        # Coba cek kalau user pakai nama lain
        if os.path.exists("bank_data_clean.csv"):
            print("Tapi ada file 'bank_data_clean.csv', pakai yang itu ya...")
            INPUT_FILE = "bank_data_clean.csv"
        else:
            exit(1) # Matikan proses kalau file gak ada

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    print("Data loaded sukses!")

    # 2. Preprocessing & Split
    # Otomatis deteksi target (jaga-jaga kalau nama kolom beda)
    if 'TransactionType' in df.columns:
        target = 'TransactionType'
    else:
        target = df.columns[-1] # Ambil kolom terakhir sebagai target

    X = df.drop(columns=[target])
    y = df[target]
    
    # Pakai data dikit aja biar GitHub Actions cepet kelar
    if len(df) > 1000:
        X = X.iloc[:1000]
        y = y.iloc[:1000]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. MLflow Tracking
    mlflow.set_experiment("Automated_CI_Run")

    with mlflow.start_run():
        print("Mulai Training...")
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, rf.predict(X_test))
        print(f"Akurasi: {acc:.4f}")

        # Log Metrics & Model (Syarat Skilled)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model_random_forest")
        
        print("Training Selesai. Artefak tersimpan.")
