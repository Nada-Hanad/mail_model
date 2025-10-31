import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import joblib
import zipfile

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ========== Load and prepare data ==========
df = pd.read_csv("mail_training_matrix_final_clean.csv")
X = df.drop(columns=["company_reference", "timestamp", "score_value", "internal_value"])
y = df["internal_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Define models to compare ==========
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        n_jobs=-1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    )
}

# ========== Directory Setup ==========
output_dir = "model_comparison_outputs"
os.makedirs(output_dir, exist_ok=True)
log_path = os.path.join(output_dir, "training_log.txt")

with open(log_path, "w") as log_file:
    log_file.write("===== Model Comparison Log =====\n\n")
    for name, model in models.items():
        log_file.write(f"--- {name} ---\n")
        print(f"Training {name}...")

        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time

        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        test_mae = mean_absolute_error(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
        mean_cv_r2 = np.mean(cv_r2)

        # Save model
        model_path = os.path.join(output_dir, f"{name}_model.joblib")
        joblib.dump(model, model_path)

        # Save predictions
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        pred_df.to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)

        # Log results
        log_file.write(f"Train MAE: {train_mae:.4f}\n")
        log_file.write(f"Train R²: {train_r2:.4f}\n")
        log_file.write(f"Test MAE: {test_mae:.4f}\n")
        log_file.write(f"Test R²: {test_r2:.4f}\n")
        log_file.write(f"CV R² scores: {np.round(cv_r2, 6)}\n")
        log_file.write(f"Mean CV R²: {mean_cv_r2:.4f}\n")
        log_file.write(f"Training time: {duration:.2f} seconds\n")
        log_file.write("\n")

# ========== Zip everything ==========
zip_name = "model_comparison_outputs.zip"
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(output_dir):
        for file in files:
            filepath = os.path.join(root, file)
            arcname = os.path.join(os.path.basename(root), file)
            zipf.write(filepath, arcname=arcname)

print(f"✅ All results saved to {zip_name}")
