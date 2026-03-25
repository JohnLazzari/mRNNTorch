import torch
from mrnntorch.mrnn.leaky_mrnn import mRNN
from typing import Tuple


class mLinearization:
    def __init__(
        self,
        rnn: mRNN,
        *args,
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

    def __call__(
        self,
        input: torch.Tensor,
        x: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
        h: torch.Tensor | None = None,
        delta_h_static: torch.Tensor | None = None,
        dh: bool = False,
    ) -> torch.Tensor:
        return self.forward(
            input, x, delta_input, delta_h, h=h, delta_h_static=delta_h_static, dh=dh
        )

    def forward(
        self,
        input: torch.Tensor,
        x: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
        h: torch.Tensor | None = None,
        delta_h_static: torch.Tensor | None = None,
        dh: bool = False,
    ) -> torch.Tensor:
        """
        First order taylor expansion of RNN at a given point and input

        Args:
            inp: 1D tensor of input for network at a given state
            h: 1D tensor of network state to linearize about
            delta_inp: perturbation of input
            delta_h: perturbation of state
        """

        # Assert correct shapes
        assert input.dim() == 1
        assert x.dim() == 1

        if h is not None:
            assert h.dim() == 1

        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians for included regions
        _jacobian, _jacobian_inp = self.jacobian(input, x, h=h, dh=dh)
        if len(self.static_region_list) >= 1:
            # Get jacobians for excluded regions if available
            _jacobian_exc, _ = self.jacobian(input, x, h, excluded_regions=True, dh=dh)
        else:
            _jacobian_exc = None

        # reshape to pass into RNN
        inp = input.unsqueeze(0).unsqueeze(0)
        x = x.unsqueeze(0)

        if h is not None:
            h = h.unsqueeze(0)

        # Get h_next for affine function
        x_next, h_next = self.rnn(inp, x, h0=h)

        out = h_next if dh else x_next

        # If there is only a single input there becomes a shape issue with squeezing
        if delta_input.shape == (1,) and _jacobian_inp.dim() == 1:
            _jacobian_inp = _jacobian_inp.unsqueeze(1)

        if _jacobian_exc is None or delta_h_static is None:
            pert = (
                out.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_inp @ delta_input.T).T
            )
        else:
            pert = (
                out.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_exc @ delta_h_static.T).T
                + (_jacobian_inp @ delta_input.T).T
            )

        return pert

    def jacobian(
        self,
        input: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        excluded_regions: bool = False,
        dh: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        assert isinstance(excluded_regions, bool)
        assert x.dim() == 1
        assert input.dim() == 1
        if h is not None:
            assert h.dim() == 1

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        input = input.unsqueeze(0).unsqueeze(0)
        x = x.unsqueeze(0)

        # Only pay attention to h if dh is true
        # if dh is False, h will be ignored
        if dh:
            assert h is not None
            h = h.unsqueeze(0)

        # For leaky mrnn, there are three inputs and two outputs
        if dh:
            _, h_jacobians = torch.autograd.functional.jacobian(self.rnn, (input, x, h))
            # unpack the tuples for x and h
            h_jacobian_input, _, h_jacobian_h = h_jacobians

            _jacobian = h_jacobian_h
            _jacobian_input = h_jacobian_input
        else:
            x_jacobians, _ = torch.autograd.functional.jacobian(self.rnn, (input, x))
            # unpack the tuples for x and h
            x_jacobian_input, x_jacobian_x = x_jacobians

            _jacobian = x_jacobian_x
            _jacobian_input = x_jacobian_input

        if excluded_regions and len(self.static_region_list) >= 1:
            excluded_to_included = []
            for r_i in self.region_list:
                excluded_to_region = []
                for r_e in self.static_region_list:
                    to_start, to_end = self.rnn.get_region_indices(r_i)
                    from_start, from_end = self.rnn.get_region_indices(r_e)
                    projection = _jacobian[to_start:to_end, from_start:from_end]
                    excluded_to_region.append(projection)
                excluded_to_region = torch.cat(excluded_to_region, dim=-1)
                excluded_to_included.append(excluded_to_region)
            _jacobian = torch.cat(excluded_to_included, dim=0)
        else:
            _jacobian = self.rnn.get_weight_subset(*self.region_list, W=_jacobian)

        return _jacobian.squeeze(), _jacobian_input.squeeze()

    def eigendecomposition(
        self,
        input: torch.Tensor,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
        dh: bool = False,
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
        _jacobian, _ = self.jacobian(input, x, h=h, dh=dh)
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
