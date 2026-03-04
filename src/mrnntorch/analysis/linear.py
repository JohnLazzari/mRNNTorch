import torch
import torch.nn.functional as F
from mrnntorch.mrnn import mRNN
from typing import Tuple


class mLinearization:
    def __init__(
        self,
        rnn: mRNN,
        *args,
        **kwargs,
    ):
        """
        Linearization object that stores methods for local analyses of mRNNs

        Args:
            mrnn: mRNN object
            W_inp: Custom input weights to be used when linearizing
            W_rec: Custom recurrent weights to be used when linearizing
        """
        self.rnn = rnn
        # Regions which are treated as grid elements
        self.zero_states = torch.zeros(
            size=(
                1,
                rnn.total_num_units,
            )
        )
        self.region_list = (
            self.rnn.hid_regions
            if not args
            else [region for region in rnn._ensure_order(*args)]
        )
        # Regions treated as static inputs for grid elements
        self.static_region_list = (
            []
            if self.region_list == self.rnn.hid_regions
            else self.rnn.get_excluded_hid_regions(*self.region_list)
        )

        # Unload kwargs
        self.W_inp = kwargs["W_inp"] if "W_inp" in kwargs else None
        self.W_rec = kwargs["W_rec"] if "W_rec" in kwargs else None

    def __call__(
        self,
        pre_activation: torch.Tensor,
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
        delta_h: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.forward(pre_activation, inp, delta_inp, delta_h, **kwargs)

    def forward(
        self,
        pre_activation: torch.Tensor,
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
        delta_h: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        First order taylor expansion of RNN at a given point and input

        Args:
            inp: 1D tensor of input for network at a given state
            h: 1D tensor of network state to linearize about
            delta_inp: perturbation of input
            delta_h: perturbation of state
        """

        # unpack kwargs
        delta_h_static = (
            kwargs["delta_h_static"] if "delta_h_static" in kwargs else None
        )

        # Assert correct shapes
        assert inp.dim() == 1
        assert pre_activation.dim() == 1

        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians for included regions
        _jacobian, _jacobian_inp = self.jacobian(pre_activation)
        if len(self.static_region_list) >= 1:
            # Get jacobians for excluded regions if available
            _jacobian_exc, _ = self.jacobian(pre_activation, excluded_regions=True)
        else:
            _jacobian_exc = None

        # reshape to pass into RNN
        inp = inp.unsqueeze(0).unsqueeze(0)
        pre_activation = pre_activation.unsqueeze(0)
        activation = self.rnn.activation(pre_activation)

        # Get h_next for affine function
        _, h_next = self.rnn(inp, pre_activation, activation)

        if _jacobian_exc is None or delta_h_static is None:
            h_pert = (
                h_next.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_inp @ delta_inp.T).T
            )
        else:
            h_pert = (
                h_next.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_exc @ delta_h_static.T).T
                + (_jacobian_inp @ delta_inp.T).T
            )

        return h_pert

    @staticmethod
    def relu_grad(x: torch.Tensor):
        """
        relu function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        # check what this returns
        return torch.autograd.functional.jacobian(F.relu, x)

    @staticmethod
    def tanh_grad(x: torch.Tensor):
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
    def sigmoid_grad(x: torch.Tensor):
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
    def softplus_grad(x: torch.Tensor):
        """
        softplus function.
        Args:
            x (torch.Tensor): pre-activation x to be used for gradient calculation
            (can be batched now)
        Returns:
            torch: Elementwise derivatives of x.
        """
        return torch.autograd.functional.jacobian(F.softplus, x)

    def jacobian(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # TODO there will be any errors in here for excluded region behavior, the shape of the matrix will be off for the affine transformation, needs to be included_region_n x excluded_region_n

        excluded_regions = (
            kwargs["excluded_regions"] if "excluded_regions" in kwargs else False
        )

        assert isinstance(excluded_regions, bool)
        assert x.dim() == 1

        if self.W_rec is None:
            # Get the subset of the weights required for jacobian
            if excluded_regions:
                if not self.static_region_list:
                    raise Exception(
                        "static region list empty, cannot gather excluded region jacobian"
                    )
                weight_subset = self.rnn.get_weight_subset(*self.static_region_list)
            else:
                weight_subset = self.rnn.get_weight_subset(*self.region_list)
        else:
            weight_subset = self.W_rec

        if self.rnn.inp_constrained:
            W_inp = self.rnn.apply_dales_law(
                self.rnn.W_inp, self.rnn.W_inp_mask, self.rnn.W_inp_sign_matrix
            )
        else:
            W_inp = self.rnn.W_inp

        # linearize the dynamics about state
        if excluded_regions:
            x_sub = self.rnn.get_region_activity(x, *self.static_region_list)
        else:
            x_sub = self.rnn.get_region_activity(x, *self.region_list)

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        # Implementing h'(x), diagonalize to multiply by W
        if self.rnn.activation_name == "relu":
            d_x_act_diag = self.relu_grad(x_sub)
        elif self.rnn.activation_name == "linear":
            d_x_act_diag = self.linear_grad(x_sub)
        elif self.rnn.activation_name == "tanh":
            d_x_act_diag = self.tanh_grad(x_sub)
        elif self.rnn.activation_name == "sigmoid":
            d_x_act_diag = self.sigmoid_grad(x_sub)
        elif self.rnn.activation_name == "softplus":
            d_x_act_diag = self.softplus_grad(x_sub)
        else:
            raise ValueError("not a valid activation function")

        # Get final jacobian using form above
        _jacobian = d_x_act_diag @ weight_subset
        _jacobian_inp = d_x_act_diag @ W_inp

        return _jacobian, _jacobian_inp

    def eigendecomposition(
        self, x: torch.Tensor, *args
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
        _jacobian, _ = self.jacobian(x, *args)
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
