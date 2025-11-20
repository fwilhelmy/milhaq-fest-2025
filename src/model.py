import merlin as ML # Package: merlinquantum, import: merlin
import torch

# Create a simple quantum layer
quantum_layer = ML.QuantumLayer.simple(
    input_size=3,
    n_params=50  # Number of trainable quantum parameters
)

# Use it like any PyTorch layer
x = torch.rand(10, 3)
output = quantum_layer(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")