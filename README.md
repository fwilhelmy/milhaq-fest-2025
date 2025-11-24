# Mil'HaQ Fest 2025 - Track 2 Swaption Forecasting
**Team: Samuel Richard · Guy-Philippe Nadon · Pacôme Gasnier · Felix Wilhelmy · Andrews A. Okine**

**Award: Best Effort Prize – Quandela/Perceval Track**

This repository contains the PercevalETS submission for Track 2 of the Mil'HaQ Fest 2025 hackathon. The challenge focuses on forecasting swaption volatility surfaces; our solution implements Quantum Long Short-Term Memory (QLSTM) models to predict future swaption values and benchmark them against a classical LSTM baseline.

## Project Overview

- **Objective:** Predict the next-day swaption volatility surface using historical surfaces (224 tenor/maturity points per day). 
- **Approach:**
  - Quantum models implemented with both gate-based and photonic QLSTMs to capture non-classical temporal dependencies.
  - Classical LSTM as a strong baseline for comparison.
  - Principal Component Analysis (PCA) optional for dimensionality reduction before model training.
- **Data:** Time-ordered swaption volatility matrices provided in `data/train.xlsx`. Dates are parsed with day-first format.

## Usage

### Environment setup

```bash
pip install -r requirements.txt
```

### Training

Use `src/main.py` to train and evaluate models. The `--model-index` flag selects the implementation:

- `0`: Photonic QLSTM
- `1`: Gate-based QLSTM (Not fully implemented..)
- `2`: Classical LSTM

Example commands:

```bash
# Train photonic QLSTM
python src/main.py --model-index 0 --max-epochs 30

# Train classical LSTM baseline
python src/main.py --data data/train.xlsx --model-index 2 --max-epochs 30
```

Key arguments:

- `--sequence-length`: Window size (days) for each input sequence.
- `--forecast-horizon`: Number of days ahead to predict.
- `--n-components`: PCA components to retain (applied to features and targets; defaults to full dimensionality).
- `--photonic-shots` / `--qlstm-shots`: Shot counts for photonic or gate-based circuits (`0` enables analytic simulation).
- `--use-photonic-head`: Use a photonic head for the photonic QLSTM output projection.
- `--use-preencoders`: Separate encoders for inputs and hidden states in the gate-based QLSTM.
- `--device-name`: PennyLane device for the gate-based model (default `default.qubit`).

Training logs and checkpoints are saved in `logs/<model-name>/` with TensorBoard summaries and the best validation checkpoints.

## Credits

- Jean Senellart's Merlin framework implementation for a QLSTM ([repository](https://github.com/jsenellart/reproduced_papers/tree/c62fc11168a890e923b8314910381559afc85314/QLSTM)).
- Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang, *Quantum Long Short-Term Memory* (arXiv:2009.01783).
