import torch
from mrnntorch.mrnn.leaky_mrnn import mRNN
from typing import Tuple
import warnings


class mLinearization:
    """Local linear analysis utilities for leaky :class:`mRNN` models."""

    def __init__(
        self,
        rnn: mRNN,
        *args,
    ):
        """Initialize the linearization helper for a model and region subset.

        Args:
            rnn (mRNN): Network to analyze.
            *args (str): Optional recurrent region names to include in the
                linearized subspace. If omitted, all recurrent regions are used.
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
        """Alias for :meth:`forward`."""
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
        """Evaluate the first-order Taylor approximation of the leaky dynamics.

        Args:
            input (torch.Tensor): External input at the operating point.
            x (torch.Tensor): Pre-activation state about which to linearize.
            delta_input (torch.Tensor): Input perturbation.
            delta_h (torch.Tensor): Perturbation for the included region subset.
            h (torch.Tensor | None): Activation corresponding to ``x`` when
                linearizing hidden activations directly.
            delta_h_static (torch.Tensor | None): Perturbation applied to excluded
                regions when only a subset of regions is linearized.
            dh (bool): If ``True``, linearize the hidden activation update instead
                of the pre-activation update.

        Returns:
            torch.Tensor: Linearized next state in the requested coordinates.
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
        """Return Jacobians of the leaky update with respect to state and input.

        Args:
            input (torch.Tensor): Input vector at which to linearize.
            x (torch.Tensor): Pre-activation state at which to linearize.
            h (torch.Tensor | None): Hidden activation used when ``dh`` is ``True``.
            excluded_regions (bool): If ``True``, return the projection from
                excluded recurrent regions into the included region subset.
            dh (bool): If ``True``, differentiate the hidden activation output
                rather than the pre-activation state output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Jacobian with respect to state
            followed by Jacobian with respect to input.
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

        if h is not None and not dh:
            warnings.warn("Provided h will be ignored since dh is False. If you want to include h, set dh to True.")

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
        """Compute the eigendecomposition of the local Jacobian.

        Args:
            input (torch.Tensor): Input vector at which to linearize.
            x (torch.Tensor): Pre-activation state at which to linearize.
            h (torch.Tensor | None): Hidden activation used when ``dh`` is ``True``.
            dh (bool): If ``True``, eigendecompose the hidden-state Jacobian.

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
