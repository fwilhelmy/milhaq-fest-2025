import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data import SwaptionDataset, load_data, prepare_data, split_data
from model import GateQLSTM, LSTM, PhotonicQLSTM


def create_dataloaders(
    data_path: Path,
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    val_size: float = 0.1,
):
    df = load_data(str(data_path))
    X, y, _ = prepare_data(df, forecast_horizon=forecast_horizon, sequence_length=sequence_length)

    feature_count = X.shape[-1]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X.reshape(-1, feature_count)).reshape(X.shape)
    y_scaled = target_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test, _, _ = split_data(X_scaled, y_scaled, np.arange(len(X_scaled)))

    train_cutoff = int(len(X_train) * (1 - val_size))
    X_val, y_val = X_train[train_cutoff:], y_train[train_cutoff:]
    X_train, y_train = X_train[:train_cutoff], y_train[:train_cutoff]

    train_loader = DataLoader(SwaptionDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SwaptionDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(SwaptionDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader, feature_count


def run_training(args):
    pl.seed_everything(args.seed)

    train_loader, val_loader, test_loader, feature_count = create_dataloaders(
        data_path=args.data,
        sequence_length=args.sequence_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
    )

    models = [
        PhotonicQLSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            shots=args.photonic_shots,
            use_photonic_head=args.use_photonic_head,
            lr=args.lr,
        ),
        GateQLSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            vqc_depth=args.vqc_depth,
            use_preencoders=args.use_preencoders,
            device_name=args.device_name,
            shots=args.qlstm_shots,
            lr=args.lr,
        ),
        LSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
        ),
    ]

    model_names = ["photonic-qlstm", "gate-qlstm", "lstm"]

    if args.model_index < 0 or args.model_index >= len(models):
        raise ValueError(
            f"model_index must be between 0 and {len(models) - 1}, got {args.model_index}"
        )

    model = models[args.model_index]

    logger = TensorBoardLogger(save_dir="logs", name=f"swaption_{model_names[args.model_index]}")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Lightning LSTM on swaption volatilities.")
    parser.add_argument("--data", type=Path, default=Path("data/train.xlsx"), help="Path to the training Excel file")
    parser.add_argument("--sequence-length", type=int, default=10, help="Sequence length for the LSTM input")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Days ahead to predict")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size of the LSTM")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-index", type=int, default=2, help="Model to train: 0=photonic-qlstm, 1=gate-qlstm, 2=lstm")
    parser.add_argument("--photonic-shots", type=int, default=0, help="Number of shots for photonic QLSTM (0=analytic)")
    parser.add_argument("--use-photonic-head", action="store_true", help="Use photonic head for photonic QLSTM output projection")
    parser.add_argument("--vqc-depth", type=int, default=2, help="Number of layers in gate-based QLSTM VQCs")
    parser.add_argument("--qlstm-shots", type=int, default=0, help="Number of shots for gate-based QLSTM circuits (0=analytic)")
    parser.add_argument("--use-preencoders", action="store_true", help="Enable separate encoders for x and h in gate-based QLSTM")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name for gate-based QLSTM")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    run_training(parser.parse_args())
