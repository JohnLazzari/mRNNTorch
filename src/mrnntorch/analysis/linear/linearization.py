import torch
import torch.nn.functional as F
import numpy as np
from mrnntorch.mRNN import mRNN


class Linearization:
    def __init__(
        self, mrnn: mRNN, W_inp: torch.Tensor = None, W_rec: torch.Tensor = None
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        Args:
            mrnn: mRNN object
            W_inp: Custom input weights to be used when linearizing
            W_rec: Custom recurrent weights to be used when linearizing
        """
        self.mrnn = mrnn
        self.W_inp = W_inp
        self.W_rec = W_rec

    @staticmethod
    def relu_grad(x: torch.Tensor) -> torch.Tensor:
        """
        relu function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.relu, x)

    @staticmethod
    def tanh_grad(x: torch.Tensor) -> torch.Tensor:
        """
        tanh function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.tanh, x)

    @staticmethod
    def sigmoid_grad(x: torch.Tensor) -> torch.Tensor:
        """
        sigmoid function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.sigmoid, x)

    @staticmethod
    def softplus_grad(x: torch.Tensor) -> torch.Tensor:
        """
        softplus function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.softplus, x)

    def jacobian(
        self, x: torch.Tensor, *args, alpha: float = 1
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Linearize the dynamics around a state and return the Jacobian.

        Computes the Jacobian of the mRNN update with respect to the hidden state
        evaluated at the provided state ``x`` and (optionally) a subset of regions
        defined by ``*args``. If ``W_inp`` is provided, also returns the Jacobian
        with respect to the input.

        Args:
            x (torch.Tensor): 1D or batched tensor representing the pre-activation state at which to
                linearize (shape ``[H]``).
            *args (str): Optional region names specifying a subset for the Jacobian.
            alpha (float): Discretization factor used in the update.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: Jacobian w.r.t. hidden
            state, and optionally (Jacobian w.r.t. input) if ``W_inp`` is provided.
        """
        assert x.dim() == 1

        if self.W_rec is None:
            # Get the subset of the weights required for jacobian
            weight_subset = self.mrnn.get_weight_subset(*args)
        else:
            weight_subset = self.W_rec

        # linearize the dynamics about state
        x_sub = self.mrnn.get_region_activity(x, *args)

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        # Implementing h'(x), diagonalize to multiply by W
        if self.mrnn.activation_name == "relu":
            d_x_act_diag = self.relu_grad(x_sub)
        elif self.mrnn.activation_name == "linear":
            d_x_act_diag = self.linear_grad(x_sub)
        elif self.mrnn.activation_name == "tanh":
            d_x_act_diag = self.tanh_grad(x_sub)
        elif self.mrnn.activation_name == "sigmoid":
            d_x_act_diag = self.sigmoid_grad(x_sub)
        elif self.mrnn.activation_name == "softplus":
            d_x_act_diag = self.softplus_grad(x_sub)
        else:
            raise ValueError("not a valid activation function")

        # Get final jacobian using form above
        _jacobian = alpha * (d_x_act_diag @ weight_subset)

        # If an input weight is specified
        if self.W_inp is not None:
            # Get final jacobian using form above
            _jacobian_inp = alpha * (d_x_act_diag @ self.W_inp)
            return _jacobian, _jacobian_inp

        return _jacobian

    def eigendecomposition(
        self, x: torch.Tensor, *args, alpha: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linearize the network and compute eigen decomposition.

        Args:
            x (torch.Tensor): 1D hidden state where the system is linearized.
            *args (str): Optional subset of regions to consider.
            alpha (float): Discretization factor.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        _jacobian = self.jacobian(x, *args, alpha=alpha)
        eigenvalues, eigenvectors = torch.linalg.eig(_jacobian)

        # Split real and imaginary parts
        reals = []
        for eigenvalue in eigenvalues:
            reals.append(eigenvalue.real.item())
        reals = torch.tensor(reals)

        ims = []
        for eigenvalue in eigenvalues:
            ims.append(eigenvalue.imag.item())
        ims = torch.tensor(ims)

        return reals, ims, eigenvectors

    def linear_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        linear function.
        Args:
            x (torch.Tensor): tensor for gradient calculation.
        Returns:
            torch: gradient of linear function.
        """
        return torch.autograd.functional.jacobian(self._linear, x)

    def _linear(self, x: torch.Tensor) -> torch.Tensor:
        return x
