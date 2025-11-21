import perceval as pcvl
import torch.nn as nn
import torch
from merlin import QuantumLayer, LexGrouping
from merlin.builder import CircuitBuilder

def prepare_quantum_model():
    # 1) Describe the circuit
    builder = CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=[0, 1, 2, 3], name="input")   # map 4 features -> 4 modes
    builder.add_rotations(trainable=True, name="theta")            # extra expressivity
    builder.add_superpositions(depth=1)                             # fixed mixing

    # 2) Turn into a trainable Torch module
    core = QuantumLayer(
        input_size=4,     # number of classical features
        builder=builder,
        n_photons=3,      # equivalent to input_state = [1,1,1,0,0,0]
        dtype=torch.float32,
    )

    # 3) Add a classical post-processing layer
    model = nn.Sequential(
        core,
        LexGrouping(core.output_size, 3),  # produces a tensor of shape (B, 3)
    )

    return model