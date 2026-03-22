"""
seeds.py — Reproducibility seeds for the dual-attention encoder-decoder pipeline.

Import this module FIRST in every script before any other imports
that involve randomness (model creation, data splitting, SHAP sampling).

Usage:
    from seeds import set_seeds
    set_seeds()          # uses default SEED = 22
    set_seeds(seed=42)   # override if needed
"""

import os
import random
import numpy as np
import tensorflow as tf


# ── The seed used to produce the results reported in the paper ────────────────
# Train MAE = 0.0159 | Val MAE = 0.0204 | Test MAE = 0.0255 | Test R² = 0.9740
PAPER_SEED = 22


def set_seeds(seed: int = PAPER_SEED) -> None:
    """
    Fix all sources of randomness to ensure reproducible results.

    Args:
        seed: Integer seed value. Use PAPER_SEED (22) to reproduce
              the exact metrics reported in the paper.
    """
    # 1. Python built-in RNG
    random.seed(seed)

    # 2. NumPy (controls SHAP background sampling, data augmentation)
    np.random.seed(seed)

    # 3. TensorFlow / Keras (weight initializers, dropout masks)
    tf.random.set_seed(seed)

    # 4. Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 5. Deterministic CUDA ops (no-op on CPU, important on GPU)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


# ── Auto-set when module is imported ─────────────────────────────────────────
set_seeds()
