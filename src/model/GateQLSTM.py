"""Gate-based Quantum LSTM components using PennyLane."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import pennylane as qml

from .BaseModel import BaseModel


# ---------------------------------------------------------------------------
# Gate-based QLSTM components (PennyLane)
# ---------------------------------------------------------------------------

def _hadamard_layer(n: int):
    for i in range(n):
        qml.Hadamard(wires=i)


def _ry_layer(weights):
    for i, w in enumerate(weights):
        qml.RY(w, wires=i)


def _entangling_layer(n: int):
    for i in range(0, n - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, n - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def _q_node(x, q_weights, n_class):
    depth = q_weights.shape[0]
    n_qubits = q_weights.shape[1]
    _hadamard_layer(n_qubits)
    x_padded = x
    if x.shape[0] < n_qubits:
        pad = torch.zeros(n_qubits - x.shape[0], dtype=x.dtype, device=x.device)
        x_padded = torch.cat([x, pad], dim=0)
    _ry_layer(x_padded)
    for _ in range(depth):
        _entangling_layer(n_qubits)
        _ry_layer(q_weights[_])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_class)]


class VQC(nn.Module):
    def __init__(self, depth: int, n_qubits: int, n_class: int, device_name: str = "default.qubit", shots: Optional[int] = None):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(depth, n_qubits))
        _shots = None if (shots is None or int(shots) <= 0) else int(shots)
        self.dev = qml.device(device_name, wires=n_qubits, shots=_shots)
        self.n_class = n_class
        self.qnode = qml.QNode(_q_node, self.dev, interface="torch")

    def forward(self, x: torch.Tensor):
        outs = []
        for sample in x:
            res = self.qnode(sample, self.weights, self.n_class)
            outs.append(torch.stack(res))
        return torch.stack(outs)


class GateBasedQuantumLSTMCell(nn.Module):
    """QLSTM cell where LSTM gates are implemented by PennyLane VQCs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        vqc_depth: int,
        device_name: str = "default.qubit",
        *,
        use_preencoders: bool = False,
        shots: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_preencoders = use_preencoders

        if use_preencoders:
            self.x_encoder = VQC(vqc_depth, n_qubits=hidden_size, n_class=hidden_size, device_name=device_name, shots=shots)
            self.h_encoder = VQC(vqc_depth, n_qubits=hidden_size, n_class=hidden_size, device_name=device_name, shots=shots)
            gate_n_qubits = 2 * hidden_size
        else:
            gate_n_qubits = input_size + hidden_size

        self.input_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.forget_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.cell_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.output_gate = VQC(vqc_depth, gate_n_qubits, hidden_size, device_name, shots=shots)
        self.output_proj = nn.Linear(hidden_size, output_size)

        logger = logging.getLogger(__name__)

        def _params(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if getattr(p, "requires_grad", False))

        total_params = _params(self)
        parts = [
            "GateBasedQuantumLSTMCell initialized:",
            f"  - device={device_name} shots={'analytic' if (shots is None or int(shots) <= 0) else int(shots)}",
            f"  - VQC depth per circuit={vqc_depth}",
            f"  - qubits per gate-VQC={gate_n_qubits}",
            f"  - pre-encoders={'enabled' if use_preencoders else 'disabled'}",
            f"  - hidden_size={hidden_size}, output_size={output_size}",
            f"  - total trainable parameters ≈ {total_params}",
        ]
        logger.info("\n".join(parts))

    def forward(self, x, state: Tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        if self.use_preencoders:
            x_enc = self.x_encoder(x)
            h_enc = self.h_encoder(h_prev)
            comb = torch.cat([x_enc, h_enc], dim=1)
        else:
            comb = torch.cat([x, h_prev], dim=1)
        i = torch.sigmoid(self.input_gate(comb))
        f = torch.sigmoid(self.forget_gate(comb))
        g = torch.tanh(self.cell_gate(comb))
        o = torch.sigmoid(self.output_gate(comb))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        out = self.output_proj(h_t)
        return out, (h_t, c_t)


class GateQLSTM(BaseModel):
    """Lightning wrapper around the gate-based QLSTM cell."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        vqc_depth: int = 2,
        *,
        use_preencoders: bool = False,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
        lr: float = 1e-3,
    ):
        super().__init__(input_size=input_size, output_size=input_size, lr=lr)
        self.hidden_size = hidden_size
        self.cell = GateBasedQuantumLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            vqc_depth=vqc_depth,
            device_name=device_name,
            use_preencoders=use_preencoders,
            shots=shots,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, timesteps, _ = x.shape
        h = torch.zeros(bsz, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        out = None
        for t in range(timesteps):
            out, (h, c) = self.cell(x[:, t, :], (h, c))
        if out is None:
            raise RuntimeError("GateQLSTM received an empty sequence")
        return out


# Backward compatibility alias
QLSTM = GateQLSTM

__all__ = ["GateBasedQuantumLSTMCell", "GateQLSTM", "QLSTM", "VQC"]
