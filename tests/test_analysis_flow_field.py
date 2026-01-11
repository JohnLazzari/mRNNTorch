"""Unit tests for FlowField container behavior.

These tests currently mark known issues (e.g., shape indexing bugs) as xfail
to highlight code paths that need fixes.
"""

import pytest
import torch

from mrnntorch.analysis.flow_fields.flow_field import FlowField


def test_flow_field_initialization_sets_state_size():
    """FlowField should derive state_size from the last grid dimension."""
    grid = torch.zeros(2, 2, 3)
    x_vels = torch.zeros(2, 2)
    y_vels = torch.zeros(2, 2)
    speeds = torch.zeros(2, 2)
    flow = FlowField(x_vels=x_vels, y_vels=y_vels, grid=grid, speeds=speeds)
    assert flow.state_size == 3


@pytest.mark.xfail(reason="FlowField sizes should not all be 3d")
def test_incorrect_shape_flow_field_initialization():
    """FlowField should derive state_size from the last grid dimension."""
    grid = torch.zeros(2, 2, 3)
    x_vels = torch.zeros(2, 2, 3)
    y_vels = torch.zeros(2, 2, 3)
    speeds = torch.zeros(2, 2)
    flow = FlowField(x_vels=x_vels, y_vels=y_vels, grid=grid, speeds=speeds)


def test_flow_field_getitem_preserves_dimensions():
    """__getitem__ should keep 3D grid structure even for int indices."""
    grid = torch.zeros(2, 2, 2)
    x_vels = torch.zeros(2, 2)
    y_vels = torch.zeros(2, 2)
    speeds = torch.zeros(2, 2)
    flow = FlowField(x_vels=x_vels, y_vels=y_vels, grid=grid, speeds=speeds)

    sub = flow[0, 1]
    assert sub.grid.shape == (1, 1, 2)
    assert sub.x_vels.shape == (1, 1)
    assert sub.y_vels.shape == (1, 1)
    assert sub.speeds.shape == (1, 1)

    sub = flow[:, 1]
    assert sub.grid.shape == (2, 1, 2)
    assert sub.x_vels.shape == (2, 1)
    assert sub.y_vels.shape == (2, 1)
    assert sub.speeds.shape == (2, 1)

    sub = flow[1, :]
    assert sub.grid.shape == (1, 2, 2)
    assert sub.x_vels.shape == (1, 2)
    assert sub.y_vels.shape == (1, 2)
    assert sub.speeds.shape == (1, 2)

    sub = flow[:, :]
    assert sub.grid.shape == (2, 2, 2)
    assert sub.x_vels.shape == (2, 2)
    assert sub.y_vels.shape == (2, 2)
    assert sub.speeds.shape == (2, 2)

    sub = flow[:]
    assert sub.grid.shape == (2, 2, 2)
    assert sub.x_vels.shape == (2, 2)
    assert sub.y_vels.shape == (2, 2)
    assert sub.speeds.shape == (2, 2)

    sub = flow[0]
    assert sub.grid.shape == (1, 2, 2)
    assert sub.x_vels.shape == (1, 2)
    assert sub.y_vels.shape == (1, 2)
    assert sub.speeds.shape == (1, 2)


def test_len():
    grid = torch.zeros(2, 2, 2)
    x_vels = torch.zeros(2, 2)
    y_vels = torch.zeros(2, 2)
    speeds = torch.zeros(2, 2)
    flow = FlowField(x_vels=x_vels, y_vels=y_vels, grid=grid, speeds=speeds)
    assert len(flow) == 4
