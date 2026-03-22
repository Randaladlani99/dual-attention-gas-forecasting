"""
shap_analysis.py — SHAP-based interpretability analysis.

Reproduces Figure 3 (SHAP summary plot + global feature importance ranking)
from Section 4 of the paper using KernelExplainer with 100 background samples.

Usage:
    python shap_analysis.py --data data/new_data_temp.csv
                            --model results/best_model.keras
"""

import argparse
import os

from seeds import set_seeds, PAPER_SEED
set_seeds()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf

from data_preparation import (
    load_data, prepare_enhanced_data,
    INPUT_STEPS, OUTPUT_STEPS
)

# ── SHAP wrapper ──────────────────────────────────────────────────────────────
def build_predict_wrapper(model, enc_steps, dec_steps, n_features):
    """
    Wraps the dual-input model into a single flat-input function
    compatible with shap.KernelExplainer.

    Input:  flat array of shape (batch, enc_steps * n_features)
    Output: flat predictions  of shape (batch, dec_steps)
    """
    def predict_wrapper(x_flat):
        batch_size = x_flat.shape[0]
        x_enc = x_flat.reshape(batch_size, enc_steps, n_features)

        # Decoder input: initialised from last encoder step
        dec_input = np.zeros((batch_size, dec_steps, n_features))
        dec_input[:, 0, :] = x_enc[:, -1, :]

        y_pred = model.predict([x_enc, dec_input], verbose=0)
        return y_pred.reshape(batch_size, -1)   # (batch, dec_steps)

    return predict_wrapper


def run_shap_analysis(
    model,
    X_train,
    X_test,
    feature_cols,
    enc_steps=INPUT_STEPS,
    dec_steps=OUTPUT_STEPS,
    n_background=100,
    save_dir="figures",
):
    """
    Compute SHAP values and produce:
        - Summary plot (beeswarm)      → figures/shap_summary.png
        - Global feature importance    → figures/shap_importance.png
        - SHAP heatmap (time × feat)   → figures/shap_heatmap.png

    Args:
        model:        Trained Keras model.
        X_train:      Training set encoder inputs (N, enc_steps, n_features).
        X_test:       Test set encoder inputs.
        feature_cols: List of feature names.
        n_background: Number of background samples for KernelExplainer.
        save_dir:     Directory to save output figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_features = len(feature_cols)

    predict_fn = build_predict_wrapper(model, enc_steps, dec_steps, n_features)

    # ── Background data (100 random training samples) ────────────────────────
    np.random.seed(PAPER_SEED)
    bg_idx = np.random.choice(len(X_train), n_background, replace=False)
    background = X_train[bg_idx].reshape(n_background, enc_steps * n_features)

    # ── Sample to explain (first test sequence) ──────────────────────────────
    sample = X_test[0:1].reshape(1, enc_steps * n_features)

    print(f"\n  Initialising KernelExplainer with {n_background} background samples...")
    explainer = shap.KernelExplainer(
        model=predict_fn,
        data=background,
        link="identity",
    )

    print("  Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(sample)   # list of (1, enc_steps*n_features)
    # each element corresponds to one decoder output step

    # ── Aggregate over decoder steps ─────────────────────────────────────────
    # shape: (enc_steps, n_features) — mean absolute SHAP over all output steps
    shap_mean = np.mean(
        [sv.reshape(enc_steps, n_features) for sv in shap_values], axis=0
    )

    # ── 1. Summary plot ───────────────────────────────────────────────────────
    print("  Generating summary plot...")
    plt.figure()
    shap.summary_plot(
        shap_mean,
        feature_names=feature_cols,
        plot_type="dot",
        max_display=16,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_dir}/shap_summary.png")

    # ── 2. Global feature importance (bar chart) ──────────────────────────────
    importance = np.abs(shap_mean).mean(axis=0)   # mean over enc_steps
    sorted_idx = np.argsort(importance)[::-1]

    plt.figure(figsize=(8, 5))
    plt.barh(
        range(n_features),
        importance[sorted_idx],
        color="#2E75B6",
    )
    plt.yticks(range(n_features), [feature_cols[i] for i in sorted_idx])
    plt.xlabel("Mean |SHAP value| (average impact on model output)")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_dir}/shap_importance.png")

    # ── 3. Heatmap (encoder time steps × features) ───────────────────────────
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        shap_mean,
        xticklabels=feature_cols,
        yticklabels=[f"t-{enc_steps - i}" for i in range(enc_steps)],
        cmap="coolwarm",
        center=0,
    )
    plt.title("SHAP Contributions — Encoder Time Steps × Features")
    plt.xlabel("Feature")
    plt.ylabel("Encoder time step")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/shap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_dir}/shap_heatmap.png")

    # ── Print ranked importance ───────────────────────────────────────────────
    print("\n  Global feature importance ranking:")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"    {rank:2d}. {feature_cols[idx]:40s} {importance[idx]:.5f}")

    return shap_values, shap_mean


def main(args):
    set_seeds(args.seed)

    df = load_data(args.data)
    (X_train, _, _,
     _,       _, _,
     X_test,  _, _,
     scalers, feature_cols) = prepare_enhanced_data(df)

    print(f"\n  Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)

    run_shap_analysis(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_cols=feature_cols,
        n_background=args.n_background,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP interpretability analysis.")
    parser.add_argument("--data",         type=str, default="data/new_data_temp.csv")
    parser.add_argument("--model",        type=str, default="results/best_model.keras")
    parser.add_argument("--n_background", type=int, default=100)
    parser.add_argument("--seed",         type=int, default=PAPER_SEED)
    args = parser.parse_args()
    main(args)
