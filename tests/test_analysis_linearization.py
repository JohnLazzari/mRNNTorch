"""Unit tests for the Linearization analysis utilities.

These tests focus on derivative helpers and Jacobian/EVD behavior while
keeping network construction minimal.
"""

import pytest
import torch

from mrnntorch.analysis.linear.linearization import Linearization
from mrnntorch.mRNN import mRNN


def test_relu_grad_returns_diagonal_jacobian():
    """ReLU autograd Jacobian should be diagonal with 0/1 entries."""
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    jac = Linearization.relu_grad(x)
    # For ReLU, derivative is 0 for x<=0 and 1 for x>0.
    expected = torch.diag(torch.tensor([0.0, 0.0, 1.0]))
    assert jac.shape == (3, 3)
    assert torch.allclose(jac, expected)


def test_sigmoid_grad_matches_autograd():
    """Sigmoid gradient should match closed-form sigma(x)*(1-sigma(x))."""
    x = torch.tensor([0.0, 1.0], requires_grad=True)
    jac = Linearization.sigmoid_grad(x)
    # Use the analytic formula for a diagonal Jacobian check.
    expected = torch.diag(torch.sigmoid(x) * (1 - torch.sigmoid(x)))
    assert torch.allclose(jac, expected)


def test_softplus_grad_positive():
    """Softplus derivative should be strictly positive for all x."""
    x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    jac = Linearization.softplus_grad(x)
    diag = torch.diagonal(jac)
    assert torch.all(diag > 0)


def test_linear_jacobian_matches_weight():
    """Linear activation yields Jacobian equal to W_rec (scaled by alpha)."""
    mrnn = mRNN(activation="linear", device="cpu")
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lin = Linearization(mrnn, W_rec=weight)
    x = torch.zeros(2, requires_grad=True)
    jac = lin.jacobian(x, alpha=0.5)
    assert torch.allclose(jac, 0.5 * weight)


def test_jacobian_returns_input_term_when_w_inp_provided():
    """If W_inp is provided, jacobian() should return (J_rec, J_inp)."""
    mrnn = mRNN(activation="linear", device="cpu")
    w_rec = torch.eye(2)
    w_inp = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    lin = Linearization(mrnn, W_rec=w_rec, W_inp=w_inp)
    x = torch.zeros(2, requires_grad=True)
    jac, jac_inp = lin.jacobian(x, alpha=1.0)
    # Both Jacobians should be returned with expected shapes and values.
    assert jac.shape == (2, 2)
    assert jac_inp.shape == (2, 3)
    assert torch.allclose(jac, w_rec)
    assert torch.allclose(jac_inp, w_inp)


def test_eigendecomposition_returns_real_imag_parts():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = mRNN(activation="linear", device="cpu")
    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    lin = Linearization(mrnn, W_rec=weight)
    x = torch.zeros(2, requires_grad=True)
    reals, ims, vecs = lin.eigendecomposition(x)
    # Eigenvectors are returned column-wise for a square matrix.
    assert vecs.shape == (2, 2)
    # Eigenvalues should be the diagonal entries for a diagonal matrix.
    assert torch.allclose(torch.sort(reals).values, torch.tensor([2.0, 3.0]))
    assert torch.allclose(ims, torch.zeros_like(ims))


def test_jacobian_requires_1d_x():
    """jacobian() asserts on non-1D inputs to avoid shape ambiguity."""
    mrnn = mRNN(activation="linear", device="cpu")
    lin = Linearization(mrnn, W_rec=torch.eye(2))
    with pytest.raises(AssertionError):
        lin.jacobian(torch.zeros(1, 2))
