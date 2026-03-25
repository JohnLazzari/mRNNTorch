import torch
from mrnntorch.analysis.linear.leaky_linear import mLinearization
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder_base import FlowFieldFinderBase
from mrnntorch.mrnn.leaky_mrnn import mRNN


class mFlowFieldFinder(FlowFieldFinderBase[mRNN]):
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
        super().__init__(
            rnn,
            fit_states,
            num_points,
            x_offset,
            y_offset,
            x_center,
            y_center,
        )
        """
        Flow field that gathers a flow field about a specified trajectory

        Args:
            mrnn (mRNN): mRNN object
            num_points (int): number of points to use in grid, results in (num_points, num_points)
            x_offset (int): scale to offset grid about trajectory in x direction
            y_offset (int): scale to offset grid about trajectory in y direction
            cancel_other_regions (bool): whether or not to zero out activity from other regions
            follow_traj (bool): whether or not to center the grid around each trajectory
        """

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
        states: torch.Tensor,
        inp: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
    ) -> list:
        """Compute 2D flow fields in a region subspace along a trajectory.

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
            stim_input = torch.zeros_like(states, dtype=self.dtype)

        # Reshape to nxd
        states, inp, stim_input = (
            self._nxd(states),
            self._nxd(inp),
            self._nxd(stim_input),
        )

        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        # get region activity for fitting and reduction
        tmp_states = self.rnn.get_region_activity(states, *self.region_list)
        reduced_traj = self._reduce_traj(tmp_states)

        if not self.static_region_list:
            # Default to dummy tensor with shape
            static_states = None
        else:
            static_states = self.rnn.get_region_activity(
                states, *self.static_region_list
            )

            if self.cancel_other_regions:
                static_states = static_states * torch.zeros_like(static_states)

        # Now going through trajectory
        for n in range(n_states):
            # default for static states
            reduced_traj_n = reduced_traj[n]
            inp_n = inp[n]
            static_states_n = static_states[n] if static_states is not None else None
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
            if static_states_n is not None:
                static_act_batch = static_states_n.repeat(low_dim_grid.shape[0], 1)
            else:
                static_act_batch = None

            full_inp_batch = inp_n.repeat(low_dim_grid.shape[0], 1)
            full_stim_batch = stim_input_n.repeat(low_dim_grid.shape[0], 1)

            # Combine the grid and static states to treat excluded regions as input
            if static_act_batch is not None:
                grid_flow = self.rnn.combine_states(
                    inverse_grid,
                    static_act_batch,
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
                    full_inp_batch.unsqueeze(self.time_dim),
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
        states: torch.Tensor,
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
        delta_h_static: torch.Tensor | None = None,
        dh: bool = False,
    ) -> list:
        """Compute linearized flow fields in a 2D subspace.

        Similar to :func:`flow_field`, but uses a local linear approximation (Jacobian)
        of the dynamics around points on the trajectory instead of a full forward
        step. Assumes no external input to the selected regions.

        Args:
            states (torch.Tensor): Hidden activations over time for selected regions, [n, d]

        Kwargs:
            delta_h (torch.Tensor): tensor containing delta of regions considered to be \
            static, i.e. regions not included in region list, which are not part of the grid

            traj_to_reduce (torch.Tensor): tensor similar to states that will be used for PCA instead of states
        Returns:
            list: FlowField objects per sampled time.
        """

        # reshape to nxd
        states, inp, delta_inp = self._nxd(states), self._nxd(inp), self._nxd(delta_inp)

        assert inp.shape[0] == delta_inp.shape[0]
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        # Activity specific to regions in region list for later computations
        region_tmp = self.rnn.get_region_activity(states, *self.region_list)
        reduced_traj = self._reduce_traj(region_tmp)

        # zero out static perturbations if regions are cancelled
        if self.cancel_other_regions and delta_h_static is not None:
            delta_h_static = delta_h_static * torch.zeros_like(delta_h_static)

        for n in range(n_states):
            states_n = states[n]
            reduced_traj_n = reduced_traj[n]
            inp_n = inp[n]
            delta_inp_n = delta_inp[n]
            delta_h_static_n = delta_h_static[n] if delta_h_static is not None else None

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
            region_states_n = self.rnn.get_region_activity(states_n, *self.region_list)
            delta_h = inverse_grid - region_states_n

            h_states_n = self.rnn.activation(states_n)

            with torch.no_grad():
                h_next = self.linearization(
                    inp_n,
                    states_n,
                    delta_inp_n,
                    delta_h,
                    h=h_states_n,
                    delta_h_static=delta_h_static_n,
                    dh=dh,
                )

            # Put next h into a grid format
            h_next = self.rnn.get_region_activity(h_next, *self.region_list)
            h_next = self._reduce_traj(h_next)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(h_next, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)

            # Reshape data back to grid
            flow_field_list.append(flow_field)

        return flow_field_list
