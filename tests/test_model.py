import pytest
import torch

from src.model import MLP


@pytest.mark.parametrize("hidden_sizes", [(), (16,), (64, 64), (8, 16, 32, 64)])
def test_any_depth_and_width_gives_one_output_per_row(hidden_sizes):
    model = MLP(in_dim=7, hidden_sizes=hidden_sizes)
    assert model(torch.randn(32, 7)).shape == (32, 1)
    assert len(model.linear_layers()) == len(hidden_sizes) + 1


def test_parameter_count():
    # 7*64 + 64  +  64*64 + 64  +  64*1 + 1
    assert MLP(7, (64, 64)).count_parameters() == 512 + 4160 + 65


def test_init_schemes_set_the_expected_weight_scale():
    torch.manual_seed(0)
    fan_in, width = 256, 256
    he = MLP(fan_in, (width,), "relu", init="he").linear_layers()[0].weight.std().item()
    xavier = MLP(fan_in, (width,), "tanh", init="xavier").linear_layers()[0].weight.std().item()
    random = MLP(fan_in, (width,), "relu", init="random").linear_layers()[0].weight.std().item()
    assert he == pytest.approx((2 / fan_in) ** 0.5, rel=0.05)              # He: var = 2 / fan_in
    assert xavier == pytest.approx((2 / (fan_in + width)) ** 0.5, rel=0.05)  # Xavier: var = 2 / (in + out)
    assert random == pytest.approx(0.01, rel=0.05)


def test_auto_init_matches_activation():
    torch.manual_seed(0)
    tanh_std = MLP(256, (256,), "tanh").linear_layers()[0].weight.std().item()
    relu_std = MLP(256, (256,), "relu").linear_layers()[0].weight.std().item()
    assert tanh_std == pytest.approx((2 / 512) ** 0.5, rel=0.05)   # Xavier for tanh
    assert relu_std == pytest.approx((2 / 256) ** 0.5, rel=0.05)   # He for ReLU
