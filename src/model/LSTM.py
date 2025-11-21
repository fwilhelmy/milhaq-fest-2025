import torch
import torch.nn as nn

from .BaseModel import BaseModel


class LSTM(BaseModel):
    """Classical LSTM baseline for swaption volatility forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
    ):
        super().__init__(input_size=input_size, output_size=input_size, lr=lr)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.output(last_hidden)
