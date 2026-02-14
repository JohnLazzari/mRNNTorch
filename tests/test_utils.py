"""Unit tests for mRNN core behaviors.

These tests focus on lightweight, CPU-safe usage patterns: initialization,
shape handling, and error/edge-case detection.
"""

import json
import numpy as np

import pytest
import torch

from mrnntorch.analysis.utils import (
    unit_vector,
    angle_between,
    line_attractor_score,
    orthogonalize,
)


def test_unit_vector():
    v1 = torch.tensor([1.0, 1.0])
    v1 = unit_vector(v1)
    assert round(torch.linalg.norm(v1).item(), 2) == 1.0


def test_angle_between():
    v1 = torch.tensor([1.0, 0.0])
    v2 = torch.tensor([0.0, 1.0])
    a = angle_between(v1, v2)
    assert a == np.radians(90)


def test_line_attractor_score():
    line_attractor_score(0.99, 0, 1)


def test_orthogonalize():
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([0.0, 1.0, 0.0])
    v3 = torch.tensor([0.0, 0.0, 1.0])
    basis = orthogonalize(v1, v2, v3)
    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            assert basis[i] @ basis[j] == 0.0
