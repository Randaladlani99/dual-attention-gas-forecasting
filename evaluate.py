"""
evaluate.py — Evaluate a trained model on train / val / test sets.

Computes MAE and R² globally and per forecasting horizon (steps 1-12),
reproducing the error analysis described in Section 4 of the paper.

Usage:
    python evaluate.py --data data/new_data_temp.csv
                       --model results/best_model.keras
"""

import argparse
import os

from seeds import set_seeds, PAPER_SEED
set_seeds()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

from data_preparation import load_data, prepare_enhanced_data


def inverse_scale(y_scaled, scaler):
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


def evaluate_split(model, X, dec, y_true, scalers, name):
    """Compute global MAE and R² for one data split."""
    y_pred = model.predict([X, dec], verbose=0)

    y_true_unscaled = inverse_scale(y_true, scalers["target_scaler"])
    y_pred_unscaled = inverse_scale(y_pred, scalers["target_scaler"])

    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    r2  = r2_score(y_true_unscaled, y_pred_unscaled)

    print(f"  {name:12s} — MAE: {mae:.4f}  |  R²: {r2:.4f}")
    return mae, r2, y_pred


def per_horizon_analysis(model, X_test, dec_test, y_test, scalers,
                         output_steps=12,
                         save_path="figures/per_horizon_mae.png"):
    """
    Compute MAE at each of the 12 forecasting steps.
    Reveals sub-linear error growth pattern discussed in Section 4.
    """
    y_pred = model.predict([X_test, dec_test], verbose=0)

    mae_per_step = []
    for step in range(output_steps):
        y_t = inverse_scale(y_test[:, step], scalers["target_scaler"])
        y_p = inverse_scale(y_pred[:, step, 0], scalers["target_scaler"])
        mae_per_step.append(mean_absolute_error(y_t, y_p))

    print("\n  Per-horizon MAE (steps 1–12):")
    for i, m in enumerate(mae_per_step, 1):
        print(f"    Step {i:2d}: {m:.4f}")

    # Plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, output_steps + 1), mae_per_step,
             marker="o", linewidth=2, color="#2E75B6")
    plt.axvline(x=6, color="red", linestyle="--", alpha=0.6,
                label="Step 6 — sub-linear growth begins")
    plt.xlabel("Forecasting Horizon (steps)")
    plt.ylabel("MAE")
    plt.title("Per-Horizon MAE on Test Set")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Per-horizon plot saved → {save_path}")
    plt.close()

    return mae_per_step


def quarterly_analysis(model, X_test, dec_test, y_test, scalers):
    """
    Compute R² by calendar quarter.
    Reproduces the seasonal drift analysis in Section 4:
        Q1: 0.9656 | Q2: 0.6260 | Q3: 0.9169 | Q4: 0.9590
    """
    y_pred = model.predict([X_test, dec_test], verbose=0)

    n = len(y_test)
    quarter_size = n // 4

    print("\n  Quarterly R² (test set):")
    for q in range(4):
        start = q * quarter_size
        end   = (q + 1) * quarter_size if q < 3 else n
        y_t = inverse_scale(y_test[start:end], scalers["target_scaler"])
        y_p = inverse_scale(y_pred[start:end], scalers["target_scaler"])
        r2  = r2_score(y_t, y_p)
        print(f"    Q{q+1}: R² = {r2:.4f}")


def main(args):
    set_seeds(args.seed)

    # Load data
    df = load_data(args.data)
    (X_train, dec_train, y_train,
     X_val,   dec_val,   y_val,
     X_test,  dec_test,  y_test,
     scalers, feature_cols) = prepare_enhanced_data(df)

    # Load model
    print(f"\n  Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    # Global metrics
    print("\n" + "=" * 55)
    print("  GLOBAL PERFORMANCE")
    print("=" * 55)
    evaluate_split(model, X_train, dec_train, y_train, scalers, "Train")
    evaluate_split(model, X_val,   dec_val,   y_val,   scalers, "Validation")
    evaluate_split(model, X_test,  dec_test,  y_test,  scalers, "Test")
    print("=" * 55)

    # Per-horizon analysis
    per_horizon_analysis(model, X_test, dec_test, y_test, scalers)

    # Quarterly analysis
    quarterly_analysis(model, X_test, dec_test, y_test, scalers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model.")
    parser.add_argument("--data",  type=str, default="data/new_data_temp.csv")
    parser.add_argument("--model", type=str, default="results/best_model.keras")
    parser.add_argument("--seed",  type=int, default=PAPER_SEED)
    args = parser.parse_args()
    main(args)
