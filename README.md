# Explainable Dual-Attention Encoder–Decoder for Natural Gas Consumption Forecasting

**Official implementation of:**

> R. Ladlani, S. Ait Taleb, A. Sebaa —
> *"Explainable Dual-Attention Encoder–Decoder Model for Natural Gas
> Consumption Forecasting Using Algerian Hourly Data"*
> ICAIABA 2026, Springer LNCS.

---

## Results

| Set | MAE | R² |
|---|---|---|
| Train | 0.0159 | 0.9915 |
| Validation | 0.0204 | 0.9715 |
| **Test** | **0.0255** | **0.9740** |

The proposed model reduces MAE by **38%** over XGBoost and **78%** over
standard LSTM. Training converges in approximately **9.5 minutes** on a
standard laptop (Intel CPU, 16 GB RAM).

---

## Architecture

The model is a BiLSTM encoder–decoder with two complementary attention mechanisms:

- **Temporal attention** (encoder): multi-head self-attention that assigns
  adaptive weights to historical time steps within the 24-hour lookback window.
- **Feature-level attention** (decoder): context attention that suppresses
  less informative features dynamically at each decoding step.

SHAP-based interpretability (KernelExplainer) quantifies feature contributions
at both local and global levels.

---

## Repository structure

```
.
├── seeds.py                  # Reproducibility seeds (SEED = 22)
├── data_preparation.py       # Data loading, feature engineering, sliding window
├── model.py                  # Dual-attention BiLSTM encoder-decoder architecture
├── train.py                  # Final model training script
├── evaluate.py               # Evaluation + per-horizon + quarterly analysis
├── shap_analysis.py          # SHAP interpretability (summary, importance, heatmap)
├── hyperparameter_search.py  # 7-configuration focused hyperparameter search
├── requirements.txt
├── data/
│   └── README.md             # Dataset access instructions
├── results/                  # Saved model weights and search results
└── figures/                  # Output plots
```

---

## Installation

```bash
git clone https://github.com/Randaladlani99/dual-attention-gas-forecasting
cd dual-attention-gas-forecasting
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare the data

See `data/README.md` for instructions on obtaining the dataset.
Place the file at `data/new_data_temp.csv`.

### 2. Train the final model

```bash
python train.py --data data/new_data_temp.csv
```

To reproduce the exact paper results, use the default seed (22):

```bash
python train.py --data data/new_data_temp.csv --seed 22
```

### 3. Evaluate

```bash
python evaluate.py --data data/new_data_temp.csv --model results/best_model.keras
```

### 4. SHAP interpretability

```bash
python shap_analysis.py --data data/new_data_temp.csv --model results/best_model.keras
```

### 5. Hyperparameter search (optional)

```bash
python hyperparameter_search.py --data data/new_data_temp.csv
```

---

## Reproducibility

All scripts import `seeds.py` before any other operation.
The seed used to produce the paper results is **SEED = 22**.

```python
from seeds import set_seeds
set_seeds()   # fixes Python, NumPy, TensorFlow, and OS hash seeds
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ladlani2026explainable,
  title     = {Explainable Dual-Attention Encoder--Decoder Model for Natural Gas
               Consumption Forecasting Using Algerian Hourly Data},
  author    = {Ladlani, Randa and Ait Taleb, Samiha and Sebaa, Abderrazak},
  booktitle = {Proceedings of ICAIABA 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

---

## Acknowledgements

The authors thank **Dr. Oussama Laib** for providing the Algerian natural gas
consumption dataset used in this study.

Meteorological data sourced from [NASA POWER](https://power.larc.nasa.gov/).

---

## License

MIT License — see `LICENSE` for details.
