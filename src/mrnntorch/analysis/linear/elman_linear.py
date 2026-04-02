import torch
from mrnntorch.mrnn.elman_mrnn import ElmanmRNN
from typing import Tuple


class emLinearization:
    """Local linear analysis utilities for :class:`ElmanmRNN` models."""

    def __init__(
        self,
        rnn: ElmanmRNN,
        *args,
    ):
        """Initialize the linearization helper for a model and region subset.

        Args:
            rnn (ElmanmRNN): Network to analyze.
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
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
        delta_h_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(
            input, h, delta_input, delta_h, delta_h_static=delta_h_static
        )

    def forward(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        delta_input: torch.Tensor,
        delta_h: torch.Tensor,
        delta_h_static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate the first-order Taylor approximation of the Elman dynamics.

        Args:
            input (torch.Tensor): External input at the operating point.
            h (torch.Tensor): Hidden state about which to linearize.
            delta_input (torch.Tensor): Input perturbation.
            delta_h (torch.Tensor): Hidden-state perturbation for included regions.
            delta_h_static (torch.Tensor | None): Perturbation applied to excluded
                regions when only a subset of regions is linearized.

        Returns:
            torch.Tensor: Linearized next hidden state.
        """

        # Assert correct shapes
        assert input.dim() == 1
        assert delta_input.dim() == 1
        assert h.dim() == 1

        if delta_h.dim() > 1:
            delta_h = delta_h.flatten(start_dim=0, end_dim=-2)

        # Get jacobians for included regions
        _jacobian, _jacobian_inp = self.jacobian(input, h)
        if len(self.static_region_list) >= 1:
            # Get jacobians for excluded regions if available
            _jacobian_exc, _ = self.jacobian(input, h, excluded_regions=True)
        else:
            _jacobian_exc = None

        # reshape to pass into RNN
        inp = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0)

        # Get h_next for affine function
        h_next = self.rnn(inp, h)

        # If there is only a single input there becomes a shape issue with squeezing
        if delta_input.shape == (1,) and _jacobian_inp.dim() == 1:
            _jacobian_inp = _jacobian_inp.unsqueeze(1)

        if _jacobian_exc is None or delta_h_static is None:
            pert = (
                h_next.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_inp @ delta_input)
            )
        else:
            pert = (
                h_next.squeeze(0)
                + (_jacobian @ delta_h.T).T
                + (_jacobian_exc @ delta_h_static)
                + (_jacobian_inp @ delta_input)
            )

        return pert

    def jacobian(
        self,
        input: torch.Tensor,
        h: torch.Tensor,
        excluded_regions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Jacobians of the Elman update with respect to state and input.

        Args:
            input (torch.Tensor): Input vector at which to linearize.
            h (torch.Tensor): Hidden state at which to linearize.
            excluded_regions (bool): If ``True``, return the projection from
                excluded recurrent regions into the included region subset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Jacobian with respect to hidden
            state followed by Jacobian with respect to input.
        """

        assert isinstance(excluded_regions, bool)
        assert h.dim() == 1
        assert input.dim() == 1

        """
            Taking jacobian of x with respect to F
            In this case, the form should be:
                J_(ij)(x) = -I_(ij) + W_(ij)h'(x_j)
        """

        input = input.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0)

        # For elman mrnn, there is a single output
        h_jacobians = torch.autograd.functional.jacobian(self.rnn, (input, h))

        # unpack the tuples for x and h
        h_jacobian_input, h_jacobian_h = h_jacobians

        _jacobian = h_jacobian_h
        _jacobian_input = h_jacobian_input

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
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the eigendecomposition of the local hidden-state Jacobian.

        Args:
            input (torch.Tensor): Input vector at which to linearize.
            h (torch.Tensor): Hidden state at which to linearize.

        Returns:
            torch.Tensor: Real parts of eigenvalues.
            torch.Tensor: Imag parts of eigenvalues.
            torch.Tensor: Eigenvectors stacked column-wise.
        """
        _jacobian, _ = self.jacobian(input, h)
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
