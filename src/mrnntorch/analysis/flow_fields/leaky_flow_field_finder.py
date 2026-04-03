import torch
from mrnntorch.analysis.linear.leaky_linear import mLinearization
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from mrnntorch.mrnn.leaky_mrnn import mRNN


class mFlowFieldFinder(FlowFieldFinderBase[mRNN]):
    """Flow-field estimator for leaky mRNN trajectories and local linearizations."""

    _default_hps = {
        "num_components": 2,
        "num_points": 50,
        "x_offset": 1,
        "y_offset": 1,
        "center": 0,
        "cancel_other_regions": False,
        "follow_traj": False,
        "name": "run",
        "dtype": torch.float32,
    }

    def __init__(
        self,
        rnn: mRNN,
        fit_states: torch.Tensor,
        num_points: int,
        x_offset: int,
        y_offset: int,
        x_center: int = 0,
        y_center: int = 0,
        follow_traj: bool = False,
        region_list: list = [],
        cancel_other_regions: bool = False,
    ):
        """Initialize a 2D flow-field finder around a trajectory.

        Args:
            rnn (mRNN): Network to analyze.
            fit_states (torch.Tensor): States used to fit the dimensionality
                reduction used for the flow-field plane.
            num_points (int): Number of grid points along each axis.
            x_offset (int): Horizontal half-width of the sampled grid.
            y_offset (int): Vertical half-width of the sampled grid.
            x_center (int): Fixed x-axis center when not following the trajectory.
            y_center (int): Fixed y-axis center when not following the trajectory.
            follow_traj (bool): If ``True``, center the grid on each sampled state.
            region_list (list): Recurrent regions to include in the reduced plane.
            cancel_other_regions (bool): If ``True``, zero activity in excluded regions.
        """
        super().__init__(
            rnn,
            fit_states,
            num_points,
            x_offset,
            y_offset,
            x_center,
            y_center,
        )

        # Unload mrnn specific kwargs
        self.cancel_other_regions = cancel_other_regions
        self.follow_traj = follow_traj

        self.zero_states = torch.zeros(
            size=(
                1,
                rnn.total_num_units,
            )
        )

        # Regions which are treated as grid elements
        self.region_list = self.rnn.hid_regions if not region_list else region_list
        # Regions treated as static inputs for grid elements
        self.static_region_list = (
            []
            if self.region_list == self.rnn.hid_regions
            else self.rnn.get_excluded_hid_regions(*self.region_list)
        )
        self.linearization = mLinearization(rnn, *self.region_list)

    def find_nonlinear_flow(
        self,
        xs: torch.Tensor,
        input: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
    ) -> list:
        """Compute nonlinear 2D flow fields in a region subspace along a trajectory.

        Projects selected region activity onto a 2D PCA subspace, constructs a grid
        around the current point, and advances the system by one step to estimate
        the local flow (velocity vectors). Can zero out non-selected regions or
        keep their control values.

        Args:
            states (torch.Tensor): Hidden activations over time [batch_size, T, N].
            inp (torch.Tensor): External input sequence.
            stim_input (torch.Tensor | None): Optional additive stimulus input.
            W (torch.Tensor | None): Optional weight matrix to use.

        Kwargs:
            stim_input (torch.Tensor): tensor input to network without weights, acts as manipulation
            W (torch.Tensor): replace the weight matrix of mRNN with W
            traj_to_reduce (torch.Tensor): tensor similar to states that will be used for PCA instead of states

        Returns:
            list: FlowField object per sampled time.
        """

        flow_field_list = []

        if stim_input is None:
            stim_input = torch.zeros_like(xs, dtype=self.dtype)

        # Reshape to nxd
        xs, input, stim_input = (
            self._nxd(xs),
            self._nxd(input),
            self._nxd(stim_input),
        )

        assert xs.shape[0] == input.shape[0]
        n_states = xs.shape[0]

        # get region activity for fitting and reduction
        tmp_states = self.rnn.get_region_activity(xs, *self.region_list)
        reduced_traj = self._reduce_traj(tmp_states)

        if not self.static_region_list:
            # Default to dummy tensor with shape
            static_xs = None
        else:
            static_xs = self.rnn.get_region_activity(xs, *self.static_region_list)

            if self.cancel_other_regions:
                static_xs = static_xs * torch.zeros_like(static_xs)

        # Now going through trajectory
        for n in range(n_states):
            # default for static states
            reduced_traj_n = reduced_traj[n]
            input_n = input[n]
            static_xs_n = static_xs[n] if static_xs is not None else None
            stim_input_n = stim_input[n]

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            if static_xs_n is not None:
                static_xs_batch = static_xs_n.repeat(low_dim_grid.shape[0], 1)
            else:
                static_xs_batch = None

            full_input_batch = input_n.repeat(low_dim_grid.shape[0], 1)
            full_stim_batch = stim_input_n.repeat(low_dim_grid.shape[0], 1)

            # Combine the grid and static states to treat excluded regions as input
            if static_xs_batch is not None:
                grid_flow = self.rnn.combine_states(
                    inverse_grid,
                    static_xs_batch,
                    self.region_list,
                    self.static_region_list,
                )
            else:
                grid_flow = inverse_grid

            # here is where we will invert the grid to get valid xs
            x_0_flow = grid_flow
            h_0_flow = self.rnn.activation(x_0_flow)

            with torch.no_grad():
                # Get activity for current timestep
                x_next, _ = self.rnn(
                    full_input_batch.unsqueeze(self.time_dim),
                    x_0_flow,
                    h_0_flow,
                    stim_input=full_stim_batch.unsqueeze(self.time_dim),
                    noise=False,
                    W_rec=W,
                )

            next_state = self.rnn.get_region_activity(x_next, *self.region_list)
            next_state_reduced = self._reduce_traj(next_state)

            x_vel, y_vel = self._compute_velocity(next_state_reduced, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)

            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self,
        xs: torch.Tensor,
        input: torch.Tensor,
        delta_input: torch.Tensor,
        delta_state_static: torch.Tensor | None = None,
    ) -> list:
        """Compute linearized 2D flow fields around sampled trajectory states.

        Similar to :func:`flow_field`, but uses a local linear approximation (Jacobian)
        of the dynamics around points on the trajectory instead of a full forward
        step. Assumes no external input to the selected regions.

        Args:
            states (torch.Tensor): Network states over time.
            inp (torch.Tensor): External input sequence aligned with ``states``.
            delta_inp (torch.Tensor): Input perturbations for the local linear model.
            delta_state_static (torch.Tensor | None): Perturbations for recurrent regions \
                excluded from the reduced plane. Should be for x or h depending on dh
            dh (bool): If ``True``, linearize hidden activations instead of
                pre-activations.

        Returns:
            list: FlowField objects per sampled time.
        """

        # reshape to nxd
        xs, input, delta_input = self._nxd(xs), self._nxd(input), self._nxd(delta_input)

        assert input.shape[0] == delta_input.shape[0]
        assert xs.shape[0] == input.shape[0]
        n_states = xs.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Activity specific to regions in region list for later computations
        region_tmp = self.rnn.get_region_activity(xs, *self.region_list)
        reduced_traj = self._reduce_traj(region_tmp)

        # zero out static perturbations if regions are cancelled
        if self.cancel_other_regions and delta_state_static is not None:
            delta_state_static = delta_state_static * torch.zeros_like(
                delta_state_static
            )

        for n in range(n_states):
            xs_n = xs[n]
            reduced_traj_n = reduced_traj[n]
            input_n = input[n]
            delta_input_n = delta_input[n]
            delta_state_static_n = (
                delta_state_static[n] if delta_state_static is not None else None
            )

            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_tv_bounds(reduced_traj_n)
                )
            else:
                lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = (
                    self._set_bounds()
                )

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            region_xs_n = self.rnn.get_region_activity(xs_n, *self.region_list)
            """
                This assumes delta state is always of x 
                This should be ok since it is a general perturbation still
            """
            delta_x = inverse_grid - region_xs_n

            with torch.no_grad():
                # get next state of h or of x if dh is false
                state_next = self.linearization(
                    input_n,
                    xs_n,
                    delta_input_n,
                    delta_x,
                    delta_state_static=delta_state_static_n,
                )

            # Put next h into a grid format
            state_next = self._reduce_traj(state_next)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(state_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)

            # Reshape data back to grid
            flow_field_list.append(flow_field)

        return flow_field_list
