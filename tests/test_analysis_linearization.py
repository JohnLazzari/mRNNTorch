"""Unit tests for the Linearization analysis utilities.

These tests focus on derivative helpers and Jacobian/EVD behavior while
keeping network construction minimal.
"""

import pytest
import torch

from mrnntorch.analysis.linear import mLinearization
from mrnntorch.mrnn import mRNN


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
    mrnn = mRNN(activation="linear", device="cpu")
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lin = mLinearization(mrnn, W_rec=weight)
    x = torch.zeros(2, requires_grad=True)
    jac = lin.jacobian(x)
    assert torch.allclose(jac, weight)


def test_eigendecomposition_returns_real_imag_parts():
    """Eigen decomposition returns real/imag parts and eigenvectors."""
    mrnn = mRNN(activation="linear", device="cpu")
    weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    lin = mLinearization(mrnn, W_rec=weight)
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
    lin = mLinearization(mrnn, W_rec=torch.eye(2))
    with pytest.raises(AssertionError):
        lin.jacobian(torch.zeros(1, 2))
