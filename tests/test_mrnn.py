"""Unit tests for mRNN core behaviors.

These tests focus on lightweight, CPU-safe usage patterns: initialization,
shape handling, and error/edge-case detection.
"""

import json

import pytest
import torch

from mrnntorch.mRNN import mRNN


def test_init_with_invalid_activation_raises():
    """Invalid activation names should fail fast with a clear exception."""
    with pytest.raises(Exception):
        mRNN(activation="not_a_real_activation")


def test_add_regions_and_indices():
    """Region indices/sizes should reflect insertion order and counts."""
    mrnn = mRNN(device="cpu")
    # Build a simple multi-region network to check indexing.
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=3, sign="neg", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_input_region(name="i1", num_units=4, sign="pos", device="cpu")

    # Total sizes should match the sum of units across regions.
    assert mrnn.total_num_units == 5
    assert mrnn.total_num_inputs == 4
    # Indices should be contiguous and ordered by definition order.
    assert mrnn.get_region_indices("r1") == (0, 2)
    assert mrnn.get_region_indices("r2") == (2, 5)
    assert mrnn.get_region_indices("i1") == (0, 4)
    assert mrnn.get_region_size("r2") == 3
    assert mrnn.get_region_size("i1") == 4


def test_get_tonic_inp_concatenates_regions():
    """Tonic input should concatenate per-region baselines in order."""
    mrnn = mRNN(device="cpu")
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0.5, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=1.5, init=0, device="cpu"
    )

    tonic = mrnn.get_tonic_inp()
    # The baseline vector length should equal total recurrent units.
    assert tonic.shape == (3,)
    assert torch.allclose(tonic, torch.tensor([0.5, 0.5, 1.5]))


def test_init_from_minimal_config(tmp_path):
    """Minimal configs should build regions even without connections."""
    config = {
        "recurrent_regions": [
            {"name": "r1", "num_units": 2, "sign": "pos", "base_firing": 0},
            {"name": "r2", "num_units": 1, "sign": "neg", "base_firing": 0},
        ],
        "input_regions": [{"name": "i1", "num_units": 3, "sign": "pos"}],
        "recurrent_connections": [],
        "input_connections": [],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    mrnn = mRNN(config=str(config_path), device="cpu")
    # Regions should be created and registered under their config names.
    assert set(mrnn.region_dict.keys()) == {"r1", "r2"}
    assert set(mrnn.inp_dict.keys()) == {"i1"}


def test_forward_shapes_without_noise_sequentially():
    """Forward pass should respect batch/time dimensions without noise."""
    mrnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    # Use a tiny network with full connectivity for deterministic shapes.
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=2, sign="neg", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.finalize_rec_connectivity()
    assert mrnn.W_rec.shape == (4, 4)

    mrnn.add_input_region(name="i1", num_units=3, sign="pos", device="cpu")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_inp_connectivity()
    assert mrnn.W_inp.shape == (4, 3)

    batch_size = 2
    seq_len = 4
    inp = torch.zeros(batch_size, seq_len, 3)
    x0 = torch.zeros(batch_size, 4)
    h0 = torch.zeros(batch_size, 4)
    xs, hs = mrnn(inp, x0, h0, noise=False)
    # Outputs should mirror [B, T, H] layout under batch_first=True.
    assert xs.shape == (batch_size, seq_len, 4)
    assert hs.shape == (batch_size, seq_len, 4)


def test_forward_shapes_without_noise_simultaneous():
    """Forward pass should respect batch/time dimensions without noise."""
    mrnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    # Use a tiny network with full connectivity for deterministic shapes.
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="neg", base_firing=0, init=0, device="cpu"
    )

    assert mrnn.region_dict["r1"].num_units == 2
    assert mrnn.region_dict["r2"].num_units == 1

    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_region(name="i1", num_units=3, sign="pos", device="cpu")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")

    assert mrnn.region_dict["r1"]["r1"].parameter.shape == (2, 2)
    assert mrnn.region_dict["r1"]["r2"].parameter.shape == (1, 2)
    assert mrnn.region_dict["r2"]["r1"].parameter.shape == (2, 1)
    assert mrnn.region_dict["r2"]["r2"].parameter.shape == (1, 1)

    assert mrnn.inp_dict["i1"]["r1"].parameter.shape == (2, 3)
    assert mrnn.inp_dict["i1"]["r2"].parameter.shape == (1, 3)

    mrnn.finalize_connectivity()

    assert mrnn.W_rec.shape == (3, 3)
    assert mrnn.W_inp.shape == (3, 3)

    batch_size = 2
    seq_len = 4
    inp = torch.zeros(batch_size, seq_len, 3)
    x0 = torch.zeros(batch_size, 3)
    h0 = torch.zeros(batch_size, 3)
    xs, hs = mrnn(inp, x0, h0, noise=False)
    # Outputs should mirror [B, T, H] layout under batch_first=True.
    assert xs.shape == (batch_size, seq_len, 3)
    assert hs.shape == (batch_size, seq_len, 3)


@pytest.mark.xfail(reason="get_region_activity calls missing __ensure_order")
def test_get_region_activity_subset_ordering():
    """Region subset ordering should be stable but currently errors."""
    mrnn = mRNN(device="cpu")
    mrnn.add_recurrent_region(
        name="r1", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    act = torch.zeros(2)
    out = mrnn.get_region_activity(act, "r2", "r1")
    assert out.shape == (2,)


@pytest.mark.xfail(reason="inp constrained init uses nn.init.torch.normal_")
def test_add_input_connection_with_constraints():
    """Constrained input init should succeed but currently calls a bad API."""
    mrnn = mRNN(device="cpu")
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_input_region(name="i1", num_units=2, sign="pos", device="cpu")
    mrnn.add_input_connection("i1", "r1")
