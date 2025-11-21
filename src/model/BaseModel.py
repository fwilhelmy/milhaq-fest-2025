import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    """Base class for swaption volatility models built with PyTorch Lightning."""

    def __init__(self, input_size: int, output_size: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError

    def _shared_step(self, batch, stage: str):
        x, y = batch
        preds = self(x)
        loss = self.criterion(preds, y)
        mae = F.l1_loss(preds, y)

        if stage != "test":
            progbar = stage == "val"
            self.log(f"loss/{stage}", loss, prog_bar=progbar, on_epoch=True, on_step=False)
            self.log(f"mae/{stage}", mae, prog_bar=progbar, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
