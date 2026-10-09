import torch

from src.utils import set_seed


def test_same_seed_gives_identical_training_losses():
    """Two training runs with the same seed must produce exactly the same losses."""

    def short_training_run(seed):
        set_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(9, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
        x, y = torch.randn(64, 9), torch.randn(64, 1)
        losses = []
        for _ in range(5):
            optimiser.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
        return losses

    assert short_training_run(seed=0) == short_training_run(seed=0)
    assert short_training_run(seed=0) != short_training_run(seed=1)

