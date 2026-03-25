"""Unit tests for FlowFieldFinder using real mRNN and PCA.

These tests focus on helper methods and shapes, while marking known issues
in higher-level flow routines as xfail.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from sklearn.decomposition import PCA

from mrnntorch.analysis import emFlowFieldFinder, mFlowFieldFinder
from mrnntorch import mRNN, ElmanmRNN


def _build_leaky_mrnn() -> mRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    return mrnn


def _build_elman_mrnn() -> ElmanmRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = ElmanmRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    return mrnn


def _build_leaky_mrnn_with_inputs() -> mRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_leaky_mrnn()
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


def _build_elman_mrnn_with_inputs() -> ElmanmRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_elman_mrnn()
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


def _sample_trajectory(batch: int = 2, seq: int = 3, units: int = 3) -> torch.Tensor:
    """Create a deterministic, non-degenerate trajectory for PCA fitting."""
    torch.manual_seed(0)
    return torch.randn(batch, seq, units)


# --------------- Test leaky flow field finder --------------------------


def test_flow_field_finder_init_sets_defaults_l():
    """Initializer should set hyperparameters and helper objects."""
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(
        mrnn,
        fit_states,
        num_points=5,
        x_offset=2,
        y_offset=3,
        cancel_other_regions=True,
        follow_traj=True,
    )
    assert finder.num_points == 5
    assert finder.x_offset == 2
    assert finder.y_offset == 3
    assert finder.cancel_other_regions is True
    assert finder.follow_traj is True
    assert isinstance(finder.reduce_obj, PCA)
    assert finder.linearization.rnn is mrnn


def test_reduce_traj_no_args_shape_l():
    """_reduce_traj should flatten [B,T,H] to [B*T,2]."""
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(mrnn, fit_states, num_points=5, x_offset=10, y_offset=10)
    trajectory = _sample_trajectory()
    finder._fit_traj(trajectory)
    reduced = finder._reduce_traj(trajectory)
    assert isinstance(reduced, torch.Tensor)
    assert reduced.shape == (trajectory.shape[0] * trajectory.shape[1], 2)


def test_inverse_grid_shapes_after_fit_l():
    """_inverse_grid should return consistent grid and inverse shapes."""
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(mrnn, fit_states, num_points=4, x_offset=10, y_offset=10)
    trajectory = _sample_trajectory()
    finder._fit_traj(trajectory)
    finder._reduce_traj(trajectory)

    low_dim_grid, inverse_grid = finder._inverse_grid(-1.0, 1.0, -2.0, 2.0)
    assert low_dim_grid.shape == (16, 2)
    assert inverse_grid.shape == (16, 3)

    low_dim_grid, inverse_grid = finder._inverse_grid(
        -1.0, 1.0, -2.0, 2.0, expand_dims=True
    )
    assert low_dim_grid.shape == (4, 4, 2)
    assert inverse_grid.shape == (4, 4, 3)


def test_compute_velocity_and_speed_normalizes_l():
    """Velocity should be elementwise diffs and speed normalized to max 1."""
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(mrnn, fit_states, num_points=10, x_offset=10, y_offset=10)

    h_prev = torch.zeros((2, 2, 2))
    h_next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    x_vel, y_vel = finder._compute_velocity(h_next, h_prev)
    assert torch.allclose(x_vel, h_next[:, :, 0])
    assert torch.allclose(y_vel, h_next[:, :, 1])
    assert isinstance(x_vel, torch.Tensor)
    assert isinstance(y_vel, torch.Tensor)

    speeds = finder._compute_speed(x_vel, y_vel)
    assert speeds.max() == 1.0
    assert isinstance(speeds, torch.Tensor)


def test_find_linear_flow_l():
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(mrnn, fit_states, num_points=3, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    inp = torch.ones(size=(1, 2, 1))
    delta_inp = torch.zeros(size=(1, 2, 1))
    finder.find_linear_flow(trajectory, inp, delta_inp)


def test_find_nonlinear_flow_l():
    mrnn = _build_leaky_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = mFlowFieldFinder(mrnn, fit_states, num_points=3, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    inp = torch.zeros(1, 2, 1)
    finder.find_nonlinear_flow(trajectory, inp)


# --------------- Test elman flow field finder --------------------------


def test_flow_field_finder_init_sets_defaults_e():
    """Initializer should set hyperparameters and helper objects."""
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(
        mrnn,
        fit_states,
        num_points=5,
        x_offset=2,
        y_offset=3,
        cancel_other_regions=True,
        follow_traj=True,
    )
    assert finder.num_points == 5
    assert finder.x_offset == 2
    assert finder.y_offset == 3
    assert finder.cancel_other_regions is True
    assert finder.follow_traj is True
    assert isinstance(finder.reduce_obj, PCA)
    assert finder.linearization.rnn is mrnn


def test_reduce_traj_no_args_shape_e():
    """_reduce_traj should flatten [B,T,H] to [B*T,2]."""
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(mrnn, fit_states, num_points=5, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory()
    finder._fit_traj(trajectory)
    reduced = finder._reduce_traj(trajectory)
    assert isinstance(reduced, torch.Tensor)
    assert reduced.shape == (trajectory.shape[0] * trajectory.shape[1], 2)


def test_inverse_grid_shapes_after_fit_e():
    """_inverse_grid should return consistent grid and inverse shapes."""
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(mrnn, fit_states, num_points=4, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory()
    finder._fit_traj(trajectory)
    finder._reduce_traj(trajectory)

    low_dim_grid, inverse_grid = finder._inverse_grid(-1.0, 1.0, -2.0, 2.0)
    assert low_dim_grid.shape == (16, 2)
    assert inverse_grid.shape == (16, 3)

    low_dim_grid, inverse_grid = finder._inverse_grid(
        -1.0, 1.0, -2.0, 2.0, expand_dims=True
    )
    assert low_dim_grid.shape == (4, 4, 2)
    assert inverse_grid.shape == (4, 4, 3)


def test_compute_velocity_and_speed_normalizes_e():
    """Velocity should be elementwise diffs and speed normalized to max 1."""
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(mrnn, fit_states, num_points=5, x_offset=5, y_offset=5)

    h_prev = torch.zeros((2, 2, 2))
    h_next = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    x_vel, y_vel = finder._compute_velocity(h_next, h_prev)
    assert torch.allclose(x_vel, h_next[:, :, 0])
    assert torch.allclose(y_vel, h_next[:, :, 1])
    assert isinstance(x_vel, torch.Tensor)
    assert isinstance(y_vel, torch.Tensor)

    speeds = finder._compute_speed(x_vel, y_vel)
    assert speeds.max() == 1.0
    assert isinstance(speeds, torch.Tensor)


def test_find_linear_flow_e():
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(mrnn, fit_states, num_points=3, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    inp = torch.ones(size=(1, 2, 1))
    delta_inp = torch.zeros(size=(1, 2, 1))
    finder.find_linear_flow(trajectory, inp, delta_inp)


def test_find_nonlinear_flow_e():
    mrnn = _build_elman_mrnn_with_inputs()
    fit_states = torch.zeros(size=(2, mrnn.total_num_units))
    finder = emFlowFieldFinder(mrnn, fit_states, num_points=3, x_offset=5, y_offset=5)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    inp = torch.zeros(1, 2, 1)
    finder.find_nonlinear_flow(trajectory, inp)
