"""Unit tests for FlowFieldFinder using real mRNN and PCA.

These tests focus on helper methods and shapes, while marking known issues
in higher-level flow routines as xfail.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from sklearn.decomposition import PCA

from mrnntorch.analysis.flow_fields.flow_field_finder import FlowFieldFinder
from mrnntorch.mRNN import mRNN


def _build_mrnn() -> mRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = mRNN(device="cpu", rec_constrained=False, inp_constrained=False)
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    return mrnn


def _build_mrnn_with_inputs() -> mRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_mrnn()
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


def test_flow_field_finder_init_sets_defaults():
    """Initializer should set hyperparameters and helper objects."""
    mrnn = _build_mrnn()
    finder = FlowFieldFinder(
        mrnn,
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
    assert finder.linearization.mrnn is mrnn


def test_reduce_traj_no_args_shape():
    """_reduce_traj should flatten [B,T,H] to [B*T,2]."""
    mrnn = _build_mrnn()
    finder = FlowFieldFinder(mrnn)
    trajectory = _sample_trajectory()
    reduced = finder._reduce_traj(trajectory)
    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (trajectory.shape[0] * trajectory.shape[1], 2)


def test_inverse_grid_shapes_after_fit():
    """_inverse_grid should return consistent grid and inverse shapes."""
    mrnn = _build_mrnn()
    finder = FlowFieldFinder(mrnn, num_points=4)
    trajectory = _sample_trajectory()
    finder._reduce_traj(trajectory)

    low_dim_grid, inverse_grid = finder._inverse_grid(-1.0, 1.0, -2.0, 2.0)
    assert low_dim_grid.shape == (16, 2)
    assert inverse_grid.shape == (16, 3)

    low_dim_grid, inverse_grid = finder._inverse_grid(
        -1.0, 1.0, -2.0, 2.0, expand_dims=True
    )
    assert low_dim_grid.shape == (4, 4, 2)
    assert inverse_grid.shape == (4, 4, 3)


def test_compute_full_trajectory_all_regions_matches_grid():
    """When all regions are included, _compute_full_trajectory should echo grid."""
    mrnn = _build_mrnn()
    finder = FlowFieldFinder(mrnn)

    grid_points = 5
    r1_vals = torch.randn(grid_points, 2)
    r2_vals = torch.randn(grid_points, 1)
    grid = torch.cat([r1_vals, r2_vals], dim=-1)
    full_act_batch = torch.randn(grid_points, 2, 3)

    region_list = list(mrnn.region_dict)
    out = finder._compute_full_trajectory(region_list, grid, full_act_batch)
    assert out.shape == grid.shape
    assert torch.allclose(out, grid)


def test_compute_velocity_and_speed_normalizes():
    """Velocity should be elementwise diffs and speed normalized to max 1."""
    mrnn = _build_mrnn()
    finder = FlowFieldFinder(mrnn)

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


def test_find_linear_flow():
    mrnn = _build_mrnn_with_inputs()
    finder = FlowFieldFinder(mrnn, num_points=3)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    finder.find_linear_flow(trajectory)


def test_find_nonlinear_flow():
    mrnn = _build_mrnn_with_inputs()
    finder = FlowFieldFinder(mrnn, num_points=3)
    trajectory = _sample_trajectory(batch=1, seq=2, units=3)
    inp = torch.zeros(1, 2, 1)
    finder.find_nonlinear_flow(trajectory, inp)
