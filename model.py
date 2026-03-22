"""
model.py — Dual-Attention BiLSTM Encoder-Decoder architecture.

Implements the proposed model from Section 3.3 of:
    "Explainable Dual-Attention Encoder-Decoder Model for Natural Gas
     Consumption Forecasting Using Algerian Hourly Data"
    R. Ladlani et al., ICAIABA 2026.

Architecture summary:
    Encoder:
        - Stacked Bidirectional LSTM (BiLSTM) with residual connection
        - Temporal multi-head self-attention
        - State projection to decoder hidden/cell states

    Decoder:
        - Feature-level multi-head context attention
        - LSTM initialized from encoder states
        - TimeDistributed Dense (swish) → TimeDistributed Dense (linear)

Loss:   Huber (delta=1.0)  — robust to consumption spikes
Optimizer: Adam
Metrics:   MAE
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Concatenate,
    Add,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


# ── Best hyperparameters reported in the paper ────────────────────────────────
BEST_CONFIG = {
    "lstm_units":    128,
    "dense_units":   32,
    "dropout_rate":  0.2,
    "num_heads":     4,
    "key_dim":       64,
    "l2_reg":        0.001,
    "learning_rate": 0.001,
}


def create_enhanced_model(
    input_shape: tuple,
    output_steps: int,
    num_features: int,
    lstm_units: int   = BEST_CONFIG["lstm_units"],
    dense_units: int  = BEST_CONFIG["dense_units"],
    dropout_rate: float = BEST_CONFIG["dropout_rate"],
    num_heads: int    = BEST_CONFIG["num_heads"],
    key_dim: int      = BEST_CONFIG["key_dim"],
    l2_reg: float     = BEST_CONFIG["l2_reg"],
    learning_rate: float = BEST_CONFIG["learning_rate"],
) -> Model:
    """
    Build and compile the dual-attention BiLSTM encoder-decoder model.

    Args:
        input_shape:    (timesteps, features) — encoder input shape.
        output_steps:   Number of future steps to predict.
        num_features:   Number of input features (= len(feature_cols)).
        lstm_units:     Hidden units for all LSTM layers.
        dense_units:    Units in the TimeDistributed intermediate dense layer.
        dropout_rate:   Dropout probability applied after LSTM layers.
        num_heads:      Number of attention heads.
        key_dim:        Key/query dimension per attention head.
        l2_reg:         L2 weight regularization coefficient.
        learning_rate:  Adam optimizer initial learning rate.

    Returns:
        Compiled Keras Model.
    """
    # ── ENCODER ──────────────────────────────────────────────────────────────
    encoder_input = Input(shape=input_shape, name="encoder_input")

    # First BiLSTM layer
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        name="bilstm_1",
    )(encoder_input)
    x = Dropout(dropout_rate)(x)

    # Second BiLSTM layer with residual connection
    x_res = Bidirectional(
        LSTM(lstm_units, return_sequences=True, kernel_regularizer=l2(l2_reg)),
        name="bilstm_2",
    )(x)
    x = Add()([x, x_res])
    x = Dropout(dropout_rate)(x)

    # Temporal multi-head self-attention (encoder)
    # — assigns adaptive weights to historical time steps
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="temporal_attention"
    )(x, x)
    attn_out = LayerNormalization()(attn_out)

    # Concatenate BiLSTM output with attention output
    encoder_out = Concatenate()([x, attn_out])

    # Project to decoder initial states
    last_step = encoder_out[:, -1, :]
    state_h = Dense(lstm_units, name="state_h")(last_step)
    state_c = Dense(lstm_units, name="state_c")(last_step)
    encoder_states = [state_h, state_c]

    # ── DECODER ──────────────────────────────────────────────────────────────
    decoder_input = Input(shape=(output_steps, num_features), name="decoder_input")

    # Feature-level context attention (decoder)
    # — learns differential importance across feature dimensions at each step
    feat_attn = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name="feature_attention"
    )(decoder_input, decoder_input)
    feat_attn = LayerNormalization()(feat_attn)

    # LSTM decoder initialized from encoder states
    decoder_lstm = LSTM(
        lstm_units, return_sequences=True, name="decoder_lstm"
    )
    decoder_out = decoder_lstm(feat_attn, initial_state=encoder_states)
    decoder_out = Dropout(dropout_rate)(decoder_out)

    # Output projection
    x = TimeDistributed(Dense(dense_units, activation="swish"))(decoder_out)
    x = Dropout(dropout_rate)(x)
    output = TimeDistributed(Dense(1), name="output")(x)

    # ── COMPILE ──────────────────────────────────────────────────────────────
    model = Model(
        inputs=[encoder_input, decoder_input],
        outputs=output,
        name="dual_attention_enc_dec",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )

    return model
