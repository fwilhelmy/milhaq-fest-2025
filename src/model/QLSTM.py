import torch
import torch.nn as nn

from merlin import QuantumLayer

from .BaseModel import BaseModel


class QuantumLSTMCell(nn.Module):
    """Quantum-inspired LSTM cell using a Merlin quantum layer for gating.

    The cell follows the classical LSTM gating mechanism but replaces the
    linear gate projections with a photonic quantum layer as described in the
    accompanying quantum LSTM notebook. Classical skip connections are kept to
    stabilize training and provide a deterministic gradient path.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_params: int = 90,
        shots: int = 0,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.shots = shots

        gate_dim = 4 * hidden_size
        quantum_input = input_size + hidden_size

        self.quantum_layer = QuantumLayer.simple(
            input_size=quantum_input,
            output_size=gate_dim,
            n_params=n_params,
            dtype=dtype,
        )
        self.quantum_projection = nn.Linear(self.quantum_layer.output_size, gate_dim)
        self.classical_skip = nn.Linear(quantum_input, gate_dim)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        gate_inputs = torch.cat([x_t, h_prev], dim=1)

        quantum_out = self.quantum_layer(gate_inputs, shots=self.shots)
        if quantum_out.dim() == 1:
            quantum_out = quantum_out.unsqueeze(0)

        gates = self.quantum_projection(quantum_out) + self.classical_skip(gate_inputs)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        input_gate = torch.sigmoid(i)
        forget_gate = torch.sigmoid(f)
        candidate = torch.tanh(g)
        output_gate = torch.sigmoid(o)

        c_t = forget_gate * c_prev + input_gate * candidate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t


class QLSTM(BaseModel):
    """Photonic quantum LSTM inspired by the Quantum_Long_Short-Term_Memory notebook."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        n_params: int = 90,
        lr: float = 1e-3,
        shots: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__(input_size=input_size, output_size=input_size, lr=lr)
        self.hidden_size = hidden_size
        self.shots = shots

        self.cell = QuantumLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            n_params=n_params,
            shots=shots,
            dtype=torch.get_default_dtype(),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h_t = x.new_zeros((batch_size, self.hidden_size))
        c_t = x.new_zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :], (h_t, c_t))
            h_t = self.dropout(h_t)

        return self.output(h_t)
