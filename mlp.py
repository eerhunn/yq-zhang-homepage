"""Reproduce the XOR experiment with PyTorch: python mlp.py."""
import torch
from torch.nn import Parameter
from torch.nn.functional import binary_cross_entropy_with_logits


def main():
    inputs = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    targets = torch.tensor([[0.], [1.], [1.], [0.]])
    weights = Parameter(torch.tensor([[2., -2.], [-2., 2.]]))
    hidden_bias = Parameter(torch.tensor([-1., -1.]))
    output_weights = Parameter(torch.tensor([[3.], [3.]]))
    output_bias = Parameter(torch.tensor([-2.5]))
    optimizer = torch.optim.Adam(
        [weights, hidden_bias, output_weights, output_bias], lr=0.03
    )

    for epoch in range(301):
        hidden = torch.sigmoid(inputs @ weights + hidden_bias)
        logits = hidden @ output_weights + output_bias
        loss = binary_cross_entropy_with_logits(logits, targets)
        if epoch in (0, 100, 200, 300):
            probabilities = logits.detach().sigmoid().flatten()
            print(f"Epoch {epoch:3d} | loss {loss.item():.6f} | p(1) {probabilities.tolist()}")
        if epoch == 300:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for name, value in (
        ("W", weights), ("b_hidden", hidden_bias),
        ("v", output_weights), ("b_output", output_bias),
    ):
        print(f"{name} = {value.detach().tolist()}")


if __name__ == "__main__":
    main()
