"""Unit tests for the Linearization analysis utilities.

These tests focus on derivative helpers and Jacobian/EVD behavior while
keeping network construction minimal.
"""

import pytest
import torch

from mrnntorch.analysis import emLinearization
from mrnntorch.analysis import mLinearization
from mrnntorch import mRNN, ElmanmRNN


def _build_leaky_mrnn(activation="relu") -> mRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = mRNN(
        device="cpu",
        rec_constrained=False,
        inp_constrained=False,
        activation=activation,
    )
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    return mrnn


def _build_leaky_mrnn_with_inputs(activation="relu") -> mRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_leaky_mrnn(activation=activation)
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


def _build_elman_mrnn(activation="relu") -> ElmanmRNN:
    """Construct a minimal mRNN with two recurrent regions on CPU."""
    mrnn = ElmanmRNN(
        device="cpu",
        rec_constrained=False,
        inp_constrained=False,
        activation=activation,
    )
    mrnn.add_recurrent_region(
        name="r1", num_units=2, sign="pos", base_firing=0, init=0, device="cpu"
    )
    mrnn.add_recurrent_region(
        name="r2", num_units=1, sign="pos", base_firing=0, init=0, device="cpu"
    )
    return mrnn


def _build_elman_mrnn_with_inputs(activation="relu") -> ElmanmRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_elman_mrnn(activation=activation)
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


# -------------- Testing leaky mRNN --------------------


def test_jacobian_matches_weight_x_l():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    lin = mLinearization(mrnn)
    x = torch.ones(3)
    inp = torch.zeros(1)
    jac, jac_inp = lin.jacobian(inp, x)
    assert torch.allclose(
        jac, (1 - mrnn.alpha) * torch.eye(3) + mrnn.alpha * mrnn.W_rec.squeeze()
    )
    assert torch.allclose(jac_inp, mrnn.alpha * mrnn.W_inp.squeeze())


def test_jacobian_matches_weight_h_l():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_leaky_mrnn_with_inputs(activation="relu")
    lin = mLinearization(mrnn)
    x = torch.tensor([1.0, 1.0, -1.0])
    h = torch.tensor([1.0, 1.0, 0.0])
    inp = torch.zeros(1)
    jac, jac_inp = lin.jacobian(inp, x, h=h, dh=True)
    assert torch.allclose(jac, torch.diag(h) @ (mrnn.alpha * mrnn.W_rec.squeeze()))
    assert torch.allclose(jac_inp, torch.diag(h) @ (mrnn.alpha * mrnn.W_inp.squeeze()))


def test_eigendecomposition_returns_real_imag_parts_l():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = _build_leaky_mrnn_with_inputs()
    lin = mLinearization(mrnn)
    x = torch.zeros(3)
    inp = torch.zeros(1)
    reals, ims, vecs = lin.eigendecomposition(inp, x)
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (3, 3)
    assert len(reals) == 3
    assert len(ims) == 3


def test_jacobian_requires_1d_x_l():
    """jacobian() asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = _build_leaky_mrnn_with_inputs()
    lin = mLinearization(mrnn)
    with pytest.raises(AssertionError):
        lin.jacobian(torch.zeros(1, 1), torch.zeros(1, 2))


# -------------- Testing elman mRNN --------------------


def test_linear_jacobian_matches_weight_e():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_elman_mrnn_with_inputs(activation="linear")
    lin = emLinearization(mrnn)
    x = torch.zeros(3)
    inp = torch.zeros(1)
    jac, jac_inp = lin.jacobian(inp, x)
    assert torch.allclose(jac, mrnn.W_rec.squeeze())
    assert torch.allclose(jac_inp, mrnn.W_inp.squeeze())


def test_eigendecomposition_returns_real_imag_parts_e():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = _build_elman_mrnn_with_inputs()
    lin = emLinearization(mrnn)
    x = torch.zeros(3)
    inp = torch.zeros(1)
    reals, ims, vecs = lin.eigendecomposition(inp, x)
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (3, 3)
    assert len(reals) == 3
    assert len(ims) == 3


def test_jacobian_requires_1d_x_e():
    """jacobian() asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = _build_elman_mrnn_with_inputs()
    lin = emLinearization(mrnn)
    with pytest.raises(AssertionError):
        lin.jacobian(torch.zeros(1, 1), torch.zeros(1, 2))
