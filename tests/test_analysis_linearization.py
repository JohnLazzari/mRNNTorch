"""Unit tests for the Linearization analysis utilities.

These tests focus on derivative helpers and Jacobian/EVD behavior while
keeping network construction minimal.
"""

import pytest
import torch

from mrnntorch.analysis.linear import mLinearization
from mrnntorch.mrnn import mRNN


def _build_mrnn(activation="relu") -> mRNN:
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


def _build_mrnn_with_inputs(activation="relu") -> mRNN:
    """Construct a minimal mRNN with input connectivity for flow calls."""
    mrnn = _build_mrnn(activation=activation)
    mrnn.add_input_region(name="i1", num_units=1, sign="pos", device="cpu")
    mrnn.add_recurrent_connection("r1", "r1")
    mrnn.add_recurrent_connection("r1", "r2")
    mrnn.add_recurrent_connection("r2", "r1")
    mrnn.add_recurrent_connection("r2", "r2")
    mrnn.add_input_connection("i1", "r1")
    mrnn.add_input_connection("i1", "r2")
    mrnn.finalize_connectivity()
    return mrnn


def test_relu_grad_returns_diagonal_jacobian():
    """ReLU autograd Jacobian should be diagonal with 0/1 entries."""
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    jac = mLinearization.relu_grad(x)
    # For ReLU, derivative is 0 for x<=0 and 1 for x>0.
    expected = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
    assert jac.shape == (3, 3)
    assert torch.allclose(jac, expected)


def test_sigmoid_grad_matches_autograd():
    """Sigmoid gradient should match closed-form sigma(x)*(1-sigma(x))."""
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    jac = mLinearization.sigmoid_grad(x)
    # Use the analytic formula for a diagonal Jacobian check.
    expected = torch.diag(torch.sigmoid(x) * (1 - torch.sigmoid(x)))
    assert torch.allclose(jac, expected)


def test_softplus_grad_positive():
    """Softplus derivative should be strictly positive for all x."""
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    jac = mLinearization.softplus_grad(x)
    diag = torch.diagonal(jac)
    assert torch.all(diag > 0)


def test_linear_jacobian_matches_weight():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = _build_mrnn_with_inputs(activation="linear")
    lin = mLinearization(mrnn)
    x = torch.zeros(3)
    jac, jac_inp = lin.jacobian(x)
    assert torch.allclose(jac, mrnn.W_rec)
    assert torch.allclose(jac_inp, mrnn.W_inp)


def test_eigendecomposition_returns_real_imag_parts():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = _build_mrnn_with_inputs()
    lin = mLinearization(mrnn)
    x = torch.zeros(3)
    reals, ims, vecs = lin.eigendecomposition(x)
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (3, 3)
    assert len(reals) == 3
    assert len(ims) == 3


def test_jacobian_requires_1d_x():
    """jacobian() asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = _build_mrnn_with_inputs()
    lin = mLinearization(mrnn)
    with pytest.raises(AssertionError):
        lin.jacobian(torch.zeros(1, 2))
