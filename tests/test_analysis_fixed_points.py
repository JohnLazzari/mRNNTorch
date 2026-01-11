"""Unit tests for FixedPointCollection utilities.

These tests validate indexing, uniqueness logic, transforms, concatenation,
and shape validation behaviors for fixed point analysis.
"""

import pytest
import torch

from mrnntorch.analysis.fixed_points.fp import FixedPointCollection


def test_fixed_point_getitem_and_setitem():
    """Indexing and assignment should preserve shapes and update data."""
    xstar = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    fps = FixedPointCollection(xstar=xstar)
    # __getitem__ with int should return a 1-row FixedPointCollection.
    subset = fps[0]
    assert subset.n == 1
    assert subset.xstar.shape == (1, 2)

    # __setitem__ should overwrite data at the given index.
    replacement = FixedPointCollection(xstar=torch.tensor([[9.0, 9.0]]))
    fps[1] = replacement
    assert torch.allclose(fps.xstar[1], torch.tensor([9.0, 9.0]))


def test_fixed_point_get_unique_prefers_smallest_qstar():
    """get_unique should keep the lowest-qstar member among duplicates."""
    xstar = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    qstar = torch.tensor([2.0, 1.0, 3.0])
    fps = FixedPointCollection(xstar=xstar, qstar=qstar)
    unique = fps.get_unique()
    # Expect two unique points: one for [0,0] (lowest qstar) and [1,1].
    assert unique.xstar.shape[0] == 2
    assert torch.allclose(unique.qstar, torch.tensor([1.0, 3.0]))
    assert torch.allclose(unique.xstar[0], torch.tensor([0.0, 0.0]))


def test_fixed_point_transform_applies_offset():
    """transform should apply a linear map and additive offset."""
    xstar = torch.tensor([[1.0, 2.0]])
    x_init = torch.tensor([[0.0, 0.0]])
    f_xstar = torch.tensor([[1.0, 1.0]])
    fps = FixedPointCollection(xstar=xstar, x_init=x_init, F_xstar=f_xstar)
    U = torch.eye(2)
    transformed = fps.transform(U, offset=1.0)
    # Offsets should apply to all transformed fields.
    assert torch.allclose(transformed.xstar, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(transformed.x_init, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(transformed.F_xstar, torch.tensor([[2.0, 2.0]]))


def test_fixed_point_concatenate():
    """concatenate should stack data across multiple collections."""
    fps1 = FixedPointCollection(xstar=torch.tensor([[0.0, 0.0]]))
    fps2 = FixedPointCollection(xstar=torch.tensor([[1.0, 1.0]]))
    combined = FixedPointCollection.concatenate([fps1, fps2])
    assert combined.xstar.shape == (2, 2)


def test_fixed_point_find_and_contains():
    """find() and __contains__ should locate matching fixed points."""
    xstar = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    inputs = torch.tensor([[0.0], [1.0]])
    fps = FixedPointCollection(xstar=xstar, inputs=inputs, tol_unique=1e-6)
    fp_single = FixedPointCollection(
        xstar=torch.tensor([[1.0, 1.0]]), inputs=torch.tensor([[1.0]])
    )
    idx = fps.find(fp_single)
    assert idx.numel() == 1
    assert int(idx[0]) == 1
    assert fp_single in fps


def test_fixed_point_update_concatenates_data():
    """update should append entries and refresh counts."""
    fps1 = FixedPointCollection(xstar=torch.tensor([[0.0, 0.0]]))
    fps2 = FixedPointCollection(xstar=torch.tensor([[1.0, 1.0]]))
    fps1.update(fps2)
    assert fps1.xstar.shape == (2, 2)


def test_fixed_point_concatenate_requires_matching_attrs():
    """concatenate should fail if nonspecific attrs mismatch."""
    fps1 = FixedPointCollection(xstar=torch.tensor([[0.0, 0.0]]), tol_unique=1e-3)
    fps2 = FixedPointCollection(xstar=torch.tensor([[1.0, 1.0]]), tol_unique=1e-4)
    with pytest.raises(AssertionError):
        FixedPointCollection.concatenate([fps1, fps2])


def test_fixed_point_assert_valid_shapes_raises():
    """Shape validation should raise on inconsistent lengths."""
    xstar = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    qstar = torch.tensor([1.0])
    with pytest.raises(AssertionError):
        FixedPointCollection(xstar=xstar, qstar=qstar)
