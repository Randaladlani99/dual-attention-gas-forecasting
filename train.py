"""
train.py — Train the dual-attention encoder-decoder model with best hyperparameters.

Reproduces the final training run reported in the paper:
    Test MAE = 0.0255 | Test R² = 0.9740
    Converged at epoch 54 in ~9.5 minutes on Intel CPU / 16 GB RAM.

Usage:
    python train.py --data data/new_data_temp.csv
    python train.py --data data/new_data_temp.csv --epochs 80 --seed 22
"""

import argparse
import os

# ── Seeds MUST be set before any other import ─────────────────────────────────
from seeds import set_seeds, PAPER_SEED

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from data_preparation import load_data, prepare_enhanced_data
from model import create_enhanced_model, BEST_CONFIG

import tensorflow as tf


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_metrics(model, X, dec_input, y_true, scalers, prefix=""):
    """Predict and compute MAE and R² on inverse-scaled values."""
    y_pred = model.predict([X, dec_input], verbose=0)

    y_true_unscaled = scalers["target_scaler"].inverse_transform(
        y_true.reshape(-1, 1)
    ).flatten()
    y_pred_unscaled = scalers["target_scaler"].inverse_transform(
        y_pred.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
    r2  = r2_score(y_true_unscaled, y_pred_unscaled)

    print(f"\n  {prefix} — MAE: {mae:.4f}  |  R²: {r2:.4f}")
    return mae, r2


# ── Learning curve plots ──────────────────────────────────────────────────────
def plot_learning_curves(history, test_mae, save_path="figures/training_curves.png"):
    """Plot MAE and Huber loss learning curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Final Model Training Analysis (Best Config)", fontweight="bold")

    # MAE
    axes[0].plot(history.history["mae"],     label="Train MAE", linewidth=2)
    axes[0].plot(history.history["val_mae"], label="Val MAE",   linewidth=2)
    axes[0].axhline(y=test_mae, color="red", linestyle="--",
                    label=f"Test MAE: {test_mae:.4f}")
    axes[0].set_title("MAE Learning Curve", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MAE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2)
    axes[1].set_title("Loss Learning Curve (Huber)", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Huber Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n  Learning curves saved → {save_path}")
    plt.close()


# ── Main training routine ─────────────────────────────────────────────────────
def main(args):
    # 1. Seeds
    set_seeds(seed=args.seed)
    print(f"\n  Seed fixed: {args.seed}")

    # 2. Load and prepare data
    print("\n  Loading data...")
    df = load_data(args.data)

    (X_train, dec_train, y_train,
     X_val,   dec_val,   y_val,
     X_test,  dec_test,  y_test,
     scalers, feature_cols) = prepare_enhanced_data(df)

    # 3. Build model
    print("\n  Building model...")
    model = create_enhanced_model(
        input_shape  = (X_train.shape[1], X_train.shape[2]),
        output_steps = y_train.shape[1],
        num_features = X_train.shape[2],
        **{k: v for k, v in BEST_CONFIG.items() if k != "learning_rate"},
        learning_rate = BEST_CONFIG["learning_rate"],
    )
    model.summary()

    # 4. Callbacks
    os.makedirs("results", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=10,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "results/best_model.keras", save_best_only=True, verbose=0
        ),
    ]

    # 5. Train
    print(f"\n  Starting training (max {args.epochs} epochs, batch {args.batch_size})...")
    history = model.fit(
        [X_train, dec_train], y_train,
        validation_data=([X_val, dec_val], y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    epochs_trained = len(history.history["loss"])
    print(f"\n  Training complete — converged at epoch {epochs_trained}.")

    # 6. Evaluate
    print("\n" + "=" * 55)
    print("  FINAL PERFORMANCE")
    print("=" * 55)
    evaluate_metrics(model, X_train, dec_train, y_train, scalers, "Train")
    evaluate_metrics(model, X_val,   dec_val,   y_val,   scalers, "Validation")
    test_mae, test_r2 = evaluate_metrics(
        model, X_test, dec_test, y_test, scalers, "Test"
    )
    print("=" * 55)
    print(f"\n  Paper target — MAE: 0.0255  |  R²: 0.9740")

    # 7. Plots
    plot_learning_curves(history, test_mae)

    print("\n  Model saved → results/best_model.keras")
    return model, history, scalers


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the dual-attention encoder-decoder model."
    )
    parser.add_argument(
        "--data", type=str, default="data/new_data_temp.csv",
        help="Path to the dataset CSV file."
    )
    parser.add_argument(
        "--epochs", type=int, default=80,
        help="Maximum number of training epochs (default: 80)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size (default: 64)."
    )
    parser.add_argument(
        "--seed", type=int, default=PAPER_SEED,
        help=f"Random seed (default: {PAPER_SEED} — reproduces paper results)."
    )
    args = parser.parse_args()
    main(args)
