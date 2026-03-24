"""
hyperparameter_search.py — Focused hyperparameter search for the dual-attention model.

Tests 7 configurations as described in Table 2 of the paper.
Results are saved to results/hyperparameter_results.csv and .json.

Usage:
    python hyperparameter_search.py --data data/new_data_temp.csv
"""

import argparse
import json
import os
from datetime import datetime

from seeds import set_seeds, PAPER_SEED
set_seeds()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score

from data_preparation import load_data, prepare_enhanced_data
from model import create_enhanced_model


# ── 7 configurations tested in the paper ─────────────────────────────────────
SEARCH_CONFIGS = [
    # Baseline (LSTM=128, 2 heads, key_dim=16, dropout=0.2)
    {"name": "baseline",        "lstm_units": 128, "num_heads": 2, "key_dim": 16,  "dropout_rate": 0.2, "dense_units": 32},
    # LSTM capacity
    {"name": "lstm_64",         "lstm_units": 64,  "num_heads": 2, "key_dim": 16,  "dropout_rate": 0.2, "dense_units": 32},
    {"name": "lstm_256",        "lstm_units": 256, "num_heads": 2, "key_dim": 16,  "dropout_rate": 0.2, "dense_units": 32},
    # Dropout
    {"name": "dropout_0.1",     "lstm_units": 128, "num_heads": 2, "key_dim": 16,  "dropout_rate": 0.1, "dense_units": 32},
    {"name": "dropout_0.3",     "lstm_units": 128, "num_heads": 2, "key_dim": 16,  "dropout_rate": 0.3, "dense_units": 32},
    # Attention capacity
    {"name": "attention_3h_32d","lstm_units": 128, "num_heads": 3, "key_dim": 32,  "dropout_rate": 0.2, "dense_units": 32},
    {"name": "attention_4h_64d","lstm_units": 128, "num_heads": 4, "key_dim": 64,  "dropout_rate": 0.2, "dense_units": 32},
    # ↑ Best config — selected for final model
]


class HyperparameterSearch:
    """Run and track hyperparameter configurations."""

    def __init__(self, X_train, dec_train, y_train,
                 X_val, dec_val, y_val, scalers):
        self.X_train   = X_train
        self.dec_train = dec_train
        self.y_train   = y_train
        self.X_val     = X_val
        self.dec_val   = dec_val
        self.y_val     = y_val
        self.scalers   = scalers
        self.results   = []

    def _unscale(self, y):
        return self.scalers["target_scaler"].inverse_transform(
            y.reshape(-1, 1)
        ).flatten()

    def _metrics(self, y_true, y_pred):
        return (
            mean_absolute_error(self._unscale(y_true), self._unscale(y_pred)),
            r2_score(self._unscale(y_true), self._unscale(y_pred)),
        )

    def test_configuration(self, config, epochs=80, batch_size=128):
        name = config.get("name", "unnamed")
        print(f"\n{'='*65}\n  Testing: {name}\n  Config: {config}\n{'='*65}")

        model = create_enhanced_model(
            input_shape  = (self.X_train.shape[1], self.X_train.shape[2]),
            output_steps = self.y_train.shape[1],
            num_features = self.X_train.shape[2],
            lstm_units   = config.get("lstm_units",   128),
            dense_units  = config.get("dense_units",  32),
            dropout_rate = config.get("dropout_rate", 0.2),
            num_heads    = config.get("num_heads",    2),
            key_dim      = config.get("key_dim",      16),
            l2_reg       = config.get("l2_reg",       0.001),
            learning_rate= config.get("learning_rate",0.001),
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae", patience=10,
                restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae", factor=0.5,
                patience=5, min_lr=1e-6, verbose=0
            ),
        ]

        t0 = datetime.now()
        history = model.fit(
            [self.X_train, self.dec_train], self.y_train,
            validation_data=([self.X_val, self.dec_val], self.y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=0,
        )
        training_sec = (datetime.now() - t0).total_seconds()

        y_pred = model.predict([self.X_val, self.dec_val], verbose=0)
        val_mae, val_r2 = self._metrics(self.y_val, y_pred)

        result = {
            "name":             name,
            "lstm_units":       config.get("lstm_units",   128),
            "num_heads":        config.get("num_heads",    2),
            "key_dim":          config.get("key_dim",      16),
            "dropout_rate":     config.get("dropout_rate", 0.2),
            "val_mae":          float(val_mae),
            "val_r2":           float(val_r2),
            "epochs_trained":   len(history.history["loss"]),
            "training_min":     round(training_sec / 60, 1),
            "total_params":     model.count_params(),
        }
        self.results.append(result)

        print(
            f"  → Val MAE: {val_mae:.4f} | R²: {val_r2:.4f} | "
            f"Epochs: {result['epochs_trained']} | "
            f"Time: {result['training_min']} min | "
            f"Params: {result['total_params']:,}"
        )

        del model
        tf.keras.backend.clear_session()
        return result

    def run(self, configs=None, epochs=80, batch_size=128):
        if configs is None:
            configs = SEARCH_CONFIGS

        print(f"\n  Starting hyperparameter search — {len(configs)} configs")
        for i, cfg in enumerate(configs, 1):
            print(f"\n  [{i}/{len(configs)}]")
            try:
                self.test_configuration(cfg, epochs=epochs, batch_size=batch_size)
            except Exception as e:
                print(f"  Config {cfg['name']} failed: {e}")

        return self.results_df()

    def results_df(self):
        return pd.DataFrame(self.results).sort_values("val_mae").reset_index(drop=True)

    def save(self, out_dir="results"):
        os.makedirs(out_dir, exist_ok=True)
        df = self.results_df()
        df.to_csv(f"{out_dir}/hyperparameter_results.csv", index=False)
        with open(f"{out_dir}/hyperparameter_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n  Results saved → {out_dir}/hyperparameter_results.csv")
        return df

    def plot(self, save_path="figures/hyperparameter_analysis.png"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = self.results_df()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Hyperparameter Search Results", fontweight="bold")

        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))

        for ax, col, label in [
            (axes[0], "val_mae", "Validation MAE"),
            (axes[1], "val_r2",  "Validation R²"),
        ]:
            ax.barh(range(len(df)), df[col], color=colors)
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["name"], fontsize=9)
            ax.set_xlabel(label)
            ax.set_title(f"{label} by Configuration", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
        plt.close()


def main(args):
    set_seeds(args.seed)

    df = load_data(args.data)
    (X_train, dec_train, y_train,
     X_val,   dec_val,   y_val,
     _,       _,         _,
     scalers, _) = prepare_enhanced_data(df)

    search = HyperparameterSearch(
        X_train, dec_train, y_train,
        X_val,   dec_val,   y_val,
        scalers,
    )
    search.run(epochs=args.epochs)
    df_res = search.save()
    search.plot()

    print("\n" + "="*65)
    print("  TOP CONFIGURATIONS")
    print("="*65)
    print(df_res[["name","val_mae","val_r2","epochs_trained",
                  "training_min","total_params"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search.")
    parser.add_argument("--data",   type=str, default="data/new_data_temp.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed",   type=int, default=PAPER_SEED)
    args = parser.parse_args()
    main(args)
