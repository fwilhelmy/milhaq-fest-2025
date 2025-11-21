import argparse
from pathlib import Path
import os
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import DeviceStatsMonitor, ModelCheckpoint

from data import SwaptionDataset, load_data, prepare_data, split_data
from model import GateQLSTM, LSTM, PhotonicQLSTM


def create_dataloaders(
    data_path: Path,
    sequence_length: int = 10,
    forecast_horizon: int = 1,
    batch_size: int = 32,
    val_size: float = 0.1,
    n_components: int | None = None,
):
    df = load_data(str(data_path))
    X, y, _ = prepare_data(df, forecast_horizon=forecast_horizon, sequence_length=sequence_length)

    feature_count = X.shape[-1]

    if n_components is None:
        n_components = feature_count

    n_components = max(1, min(n_components, feature_count))

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    X_scaled = feature_scaler.fit_transform(X.reshape(-1, feature_count)).reshape(X.shape)
    y_scaled = target_scaler.fit_transform(y)

    if n_components < feature_count:
        pca = PCA(n_components=n_components)

        flattened_X = X_scaled.reshape(-1, feature_count)
        X_scaled = pca.fit_transform(flattened_X).reshape(X.shape[0], X.shape[1], n_components)

        y_scaled = pca.transform(y_scaled)
        feature_count = n_components

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
        n_components=args.n_components,
    )

    if args.model_index==0:
        print("Training Photonic QLSTM Model")
        model = PhotonicQLSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            shots=args.photonic_shots,
            use_photonic_head=args.use_photonic_head,
            lr=args.lr,
        )
    elif args.model_index==1:
        print("Training Gate-based QLSTM Model")
        model =GateQLSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            vqc_depth=args.vqc_depth,
            use_preencoders=args.use_preencoders,
            device_name=args.device_name,
            shots=args.qlstm_shots,
            lr=args.lr,
        )
    else:
        print("Training Classical LSTM Model")
        model = LSTM(
            input_size=feature_count,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
        )

    model_names = ["photonic-qlstm", "gate-qlstm", "lstm"]

    if args.model_index < 0 or args.model_index >= len(model_names):
        raise ValueError(
            f"model_index must be between 0 and {len(model_names) - 1}, got {args.model_index}"
        )

    log_dir = "logs"
    model_dir = os.path.join(log_dir, model_names[args.model_index])
    os.makedirs(model_dir, exist_ok=True)
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch={epoch:02d}-loss={val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=3,
        save_on_train_epoch_end=False,
        monitor="val/loss",
        mode="min"
    )

    logger = TensorBoardLogger(save_dir="logs", name=f"swaption_{model_names[args.model_index]}")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb],
        log_every_n_steps=10,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    return model, model_names[args.model_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Lightning LSTM on swaption volatilities.")
    parser.add_argument("--data", type=Path, default=Path("data/train.xlsx"), help="Path to the training Excel file")
    parser.add_argument("--sequence-length", type=int, default=10, help="Sequence length for the LSTM input")
    parser.add_argument("--forecast-horizon", type=int, default=1, help="Days ahead to predict")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--hidden-size", type=int, default=4, help="Hidden size of the LSTM")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout between LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-index", type=int, default=0, help="Model to train: 0=photonic-qlstm, 1=gate-qlstm, 2=lstm")
    parser.add_argument("--photonic-shots", type=int, default=0, help="Number of shots for photonic QLSTM (0=analytic)")
    parser.add_argument("--use-photonic-head", action="store_true", help="Use photonic head for photonic QLSTM output projection")
    parser.add_argument("--vqc-depth", type=int, default=2, help="Number of layers in gate-based QLSTM VQCs")
    parser.add_argument("--qlstm-shots", type=int, default=0, help="Number of shots for gate-based QLSTM circuits (0=analytic)")
    parser.add_argument("--use-preencoders", action="store_true", help="Enable separate encoders for x and h in gate-based QLSTM")
    parser.add_argument("--device-name", type=str, default="default.qubit", help="PennyLane device name for gate-based QLSTM")
    parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        help="Number of PCA components to retain (applied to both features and targets)",
    )

    run_training(parser.parse_args())
