"""Unit tests for FixedPointFinder helpers and input handling."""

import numpy as np
import pytest
import torch

from mrnntorch.analysis.fixed_points.fp import FixedPointCollection
from mrnntorch.analysis.fixed_points.fp_finder import FixedPointFinder
from mrnntorch.mRNN import mRNN


def _build_mrnn() -> mRNN:
    """Construct a minimal mRNN with recurrent + input connectivity on CPU."""
    mrnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    mrnn.add_recurrent_region(
        name="r1", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_input_connection("i1", "r1")
    mrnn.finalize_connectivity()
    return mrnn


def test_fixed_point_finder_default_hps_copy():
    """default_hps should return a deep copy of defaults."""
    hps = FixedPointFinder.default_hps()
    hps["lr_init"] = 123.0
    assert FixedPointFinder.default_hps()["lr_init"] != 123.0


def test_sample_states_excludes_zero_tensors():
    """sample_states should avoid rows that are entirely zero."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(mrnn, verbose=False)
    torch.manual_seed(0)
    state_traj = torch.tensor([[[0.0], [1.0]], [[0.0], [2.0]]])
    samples = finder.sample_states(state_traj, n_inits=3, exclude_zero_tensors=True)
    assert samples.shape == (3, 1)
    assert torch.all(samples != 0).item()


def test_sample_states_negative_noise_raises():
    """Negative noise_scale should raise ValueError."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(mrnn, verbose=False)
    state_traj = torch.ones((1, 1, 1))
    with pytest.raises(ValueError):
        finder.sample_states(state_traj, n_inits=1, noise_scale=-0.1)


def test_identify_q_outliers_and_non_outliers():
    """Outlier helpers should split indices by threshold."""
    xstar = torch.zeros((3, 1))
    qstar = torch.tensor([0.5, 2.0, 1.5])
    fps = FixedPointCollection(xstar=xstar, qstar=qstar)
    outliers = FixedPointFinder.identify_q_outliers(fps, q_thresh=1.0)
    non_outliers = FixedPointFinder.identify_q_non_outliers(fps, q_thresh=1.0)
    assert torch.equal(outliers, torch.tensor([1, 2]))
    assert torch.equal(non_outliers, torch.tensor([0]))


def test_distance_outlier_helpers():
    """Distance helpers should flag points beyond threshold."""
    initial_states = torch.tensor([[0.0], [1.0], [10.0]])
    init_idx = FixedPointFinder.get_init_non_distance_outliers(
        initial_states, dist_thresh=1.0
    )
    assert torch.equal(init_idx, torch.tensor([0, 1]))

    fps = FixedPointCollection(xstar=torch.tensor([[0.0], [10.0]]))
    fps_idx = FixedPointFinder.get_fp_non_distance_outliers(
        fps, initial_states, dist_thresh=1.0
    )
    assert torch.equal(fps_idx, torch.tensor([0]))


def test_exclude_distance_outliers_filters_fps():
    """_exclude_distance_outliers should drop faraway fixed points."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(
        mrnn,
        outlier_distance_scale=1.0,
        do_exclude_distance_outliers=True,
        verbose=False,
    )
    initial_states = torch.tensor([[0.0], [1.0], [10.0]])
    fps = FixedPointCollection(xstar=initial_states.clone())
    filtered = finder._exclude_distance_outliers(fps, initial_states)
    assert filtered.n == 2
    assert torch.allclose(filtered.xstar.squeeze(), torch.tensor([0.0, 1.0]))


def test_broadcast_tiles_inputs_and_defaults(monkeypatch):
    """find_fixed_points should tile single inputs and pass default stim."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(mrnn, do_exclude_distance_outliers=False, verbose=False)

    initial_states = torch.tensor([[[0.0], [1.0], [2.0]]])
    ext_inputs = torch.tensor([1.0])
    stim_inputs = torch.tensor([0.0])

    initial_states = finder._broadcast_nxd(initial_states, tile_n=1)
    n = initial_states.shape[0]
    ext_inputs = finder._broadcast_nxd(ext_inputs, tile_n=n)
    stim_inputs = finder._broadcast_nxd(stim_inputs, tile_n=n)

    assert initial_states.shape == (3, 1)
    assert torch.allclose(ext_inputs, torch.ones((3, 1)))
    assert torch.allclose(stim_inputs, torch.zeros((3, 1)))


def test_find_fixed_points_rejects_bad_input_shape():
    """find_fixed_points should reject incompatible input batch sizes."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(mrnn, do_exclude_distance_outliers=False, verbose=False)
    initial_states = torch.zeros((2, 1))
    ext_inputs = torch.zeros((3, 1))
    with pytest.raises(AssertionError):
        finder.find_fixed_points(initial_states, ext_inputs)


def test_fp_optimization_smoke():
    """_fp_optimization should return a populated FixedPointCollection."""
    mrnn = _build_mrnn()
    finder = FixedPointFinder(mrnn, max_iters=1, verbose=False, super_verbose=False)
    initial_states = torch.zeros((2, 1), dtype=torch.float32)
    ext_inputs = torch.zeros((2, 1, 1), dtype=torch.float32)
    stim_inp = torch.zeros((2, 1, 1), dtype=torch.float32)

    fps = finder._fp_optimization(initial_states, ext_inputs, stim_inp=stim_inp)

    assert isinstance(fps, FixedPointCollection)
    assert fps.n == 2
    assert fps.xstar.shape == (2, 1)
    assert fps.inputs.shape == (2, 1)
    assert fps.F_xstar.shape == (2, 1)
