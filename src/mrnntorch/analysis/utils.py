import numpy as np
import torch
from typing import Tuple


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Returns the unit vector of the vector."""
    return vector / torch.linalg.norm(vector)


def angle_between(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Returns the angle in radians between vectors 'v1' and 'v2'::"""
    if torch.all(v1 == 0):
        v1_u = v1
    else:
        v1_u = unit_vector(v1)

    if torch.all(v2 == 0):
        v2_u = v2
    else:
        v2_u = unit_vector(v2)
    return torch.arccos(v1_u @ v2_u)


def line_attractor_score(
    lambda_1: float, lambda_2: float, tau: float
) -> Tuple[float, float, float]:
    """
    Calculate the line attractor score based on the eigenvalues (lambda_1, lambda_2)
    and the time constant (tau).

    Args:
        lambda_1 (float): First eigenvalue.
        lambda_2 (float): Second eigenvalue.
        tau (float): Time constant.

    Returns:
        float: Calculated line attractor score.
    """
    lambd_1_dist = abs(1 - lambda_1)
    lambd_2_dist = abs(1 - lambda_2)
    tau_1 = tau / lambd_1_dist
    tau_2 = tau / lambd_2_dist
    score = np.log2(tau_1 / tau_2)
    return score, lambd_1_dist, lambd_2_dist


def projection(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Projection of tensor v1 onto tensor v2
    """
    return torch.dot(v1, v2) / torch.dot(v2, v2) * v2


def orthogonalize(v1: torch.Tensor, *args) -> Tuple[torch.Tensor, ...]:
    """Find orthgonal basis for passed in LDA objects
    This function will update the mode for each passed in object
    according to the new basis

    Args:
        v1 (torch.Tensor): an initial vector to begin orthgonalization
        args: any additional number of vectors
    """
    v1 = v1
    orth_vecs = (v1,)
    for v in args:
        sub_projection = v.clone()
        projections = [projection(v, orth_vec) for orth_vec in orth_vecs]
        sub_projection = projections[0] - torch.stack(projections[1:]).sum(dim=0)
        orth_vecs = (*orth_vecs, sub_projection)
    return orth_vecs
