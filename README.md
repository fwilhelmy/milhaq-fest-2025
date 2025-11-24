# Mil'HaQ Fest 2025 - Track 2 Swaption Forecasting

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Training

Use `src/main.py` to train and evaluate models. The `--model-index` flag selects the implementation:

- `0`: Photonic QLSTM (`PhotonicQLSTM`)
- `1`: Gate-based QLSTM (`GateQLSTM`)
- `2`: Classical LSTM (`LSTM`)

Example commands:

```bash
# Train photonic QLSTM (analytic mode)
python src/main.py --data data/train.xlsx --model-index 0 --sequence-length 10 --forecast-horizon 1 --max-epochs 10

# Train gate-based QLSTM with variational depth 3
python src/main.py --data data/train.xlsx --model-index 1 --vqc-depth 3 --hidden-size 8 --sequence-length 10 --forecast-horizon 1

# Train classical LSTM baseline
python src/main.py --data data/train.xlsx --model-index 2 --hidden-size 32 --num-layers 2 --dropout 0.1 --max-epochs 20
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

- Quantum LSTM implementation inspiration: Jean Senellart's Merlin framework example for QLSTM ([repository](https://github.com/jsenellart/reproduced_papers/tree/c62fc11168a890e923b8314910381559afc85314/QLSTM)).
- Research paper: Samuel Yen-Chi Chen, Shinjae Yoo, and Yao-Lung L. Fang, *Quantum Long Short-Term Memory* (arXiv:2009.01783).
- Team: Samuel Richard, Guy-Philippe Nadon, Pacôme Gasnier, Felix Wilhelmy, and Andrews A. Okine.

