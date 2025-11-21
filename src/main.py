import torch
import torch.nn.functional as F

from data import prepare_data
from model import prepare_quantum_model


def run_experiment(layer: torch.nn.Module, epochs: int = 80, lr: float = 5e-2):
    X_train, X_test, y_train, y_test = prepare_data()

    opt = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(epochs):
        layer.train()
        opt.zero_grad()
        logits = layer(X_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        opt.step()

    layer.eval()
    with torch.no_grad():
        train_acc = (layer(X_train).argmax(1) == y_train).float().mean().item()
        test_acc  = (layer(X_test).argmax(1) == y_test).float().mean().item()
    return train_acc, test_acc

if __name__ == "__main__":
    model = prepare_quantum_model()
    train_acc, test_acc = run_experiment(model, epochs=80, lr=0.05)
    print(f"Train acc: {train_acc:.3f}  |  Test acc: {test_acc:.3f}")