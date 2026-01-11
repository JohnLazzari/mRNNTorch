"""Unit tests for region creation and connection behavior.

These tests exercise region-level building blocks (recurrent/input) and
basic mRNN wiring assumptions without invoking heavy training loops.
"""

import torch

from mrnntorch.mRNN import mRNN
from mrnntorch.region.input_region import InputRegion
from mrnntorch.region.recurrent_region import RecurrentRegion


def test_add_recurrent_region():
    """Adding a recurrent region populates region dict and mask metadata."""
    mrnn = mRNN(device="cpu")
    # Use a small region to keep test runtime and memory minimal.
    params = {
        "name": "reg",
        "num_units": 4,
        "sign": "pos",
        "base_firing": 1,
        "init": 1,
        "device": "cpu",
        "parent_region": "reg",
        "learnable_bias": True,
    }
    mrnn.add_recurrent_region(**params)
    # Region is registered and total unit count is updated.
    assert "reg" in mrnn.region_dict
    assert mrnn.total_num_units == 4
    # Region mask should be full ones because there is only a single region.
    mask = mrnn.region_mask_dict["reg"]
    assert mask.shape == (4,)
    assert torch.all(mask == 1)


def test_add_input_region():
    """Adding an input region updates input registry and unit count."""
    mrnn = mRNN(device="cpu")
    mrnn.add_input_region(name="inp", num_units=3, sign="neg", device="cpu")
    assert "inp" in mrnn.inp_dict
    assert mrnn.total_num_inputs == 3


def test_recurrent_region_learnable_bias():
    """RecurrentRegion honors learnable_bias by using nn.Parameter."""
    region = RecurrentRegion(
        num_units=2, base_firing=0.5, learnable_bias=True, device="cpu"
    )
    assert isinstance(region.base_firing, torch.nn.Parameter)
    assert region.base_firing.shape == (2,)


def test_input_region_zero_connection_masks():
    """Zero connections must zero out masks to prevent any effect."""
    src = InputRegion(num_units=2, sign="pos", device="cpu")
    dst = RecurrentRegion(num_units=3, sign="pos", device="cpu")
    src.add_connection(
        dst_region_name="dst",
        dst_region_units=dst.num_units,
        sparsity=None,
        zero_connection=True,
    )
    connection = src.connections["dst"]
    # Zero connections should enforce fully zero masks and sign matrices.
    assert torch.all(connection.weight_mask == 0)
    assert torch.all(connection.sign_matrix == 0)


def test_input_region_sign_matrix_respects_sign():
    """Sign masks for input regions must match the region sign."""
    src = InputRegion(num_units=2, sign="neg", device="cpu")
    dst = RecurrentRegion(num_units=3, sign="pos", device="cpu")
    src.add_connection(
        dst_region_name="dst", dst_region_units=dst.num_units, sparsity=None
    )
    connection = src.connections["dst"]
    # Dense mask yields all ones and sign should be negative for neg regions.
    assert torch.all(connection.weight_mask == 1)
    assert torch.all(connection.sign_matrix == -1)
