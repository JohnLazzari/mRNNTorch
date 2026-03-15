import torch
from mrnntorch.analysis.linear import mLinearization
from sklearn.decomposition import PCA
from rnntoolkit.flow_fields.flow_field import FlowField
from rnntoolkit.flow_fields.flow_field_finder import FlowFieldFinder
from mrnntorch.mrnn.leaky_mrnn import mRNN


class mFlowFieldFinder(FlowFieldFinder[mRNN]):
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
        *args,
        **kwargs,
    ):
        super().__init__(rnn, **kwargs)
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
        self.cancel_other_regions = (
            kwargs["cancel_other_regions"]
            if "cancel_other_regions" in kwargs
            else self._default_hps["cancel_other_regions"]
        )

        self.zero_states = torch.zeros(
            size=(
                1,
                rnn.total_num_units,
            )
        )

        # Regions which are treated as grid elements
        self.region_list = (
            self.rnn.hid_regions
            if not args
            else [region for region in self.rnn._ensure_order(*args)]
        )
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
        **kwargs,
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

        # Unloading Kwargs
        stim_input = kwargs["stim_input"] if "stim_input" in kwargs else None
        W = kwargs["W"] if "W" in kwargs else None
        traj_to_reduce = (
            kwargs["traj_to_reduce"] if "traj_to_reduce" in kwargs else states
        )

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

        """
        states is now meant to be network activation, or h.
        To get valid xs, we will invert h in the compute function
        """

        # get region activity for fitting and reduction
        tmp_states_to_reduce = self.rnn.get_region_activity(
            traj_to_reduce, *self.region_list
        )
        tmp_states = self.rnn.get_region_activity(states, *self.region_list)

        self._fit_traj(tmp_states_to_reduce)
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
            static_states_n = static_states[n] if static_states is not None else None
            # call nonlinear flowfield computation
            flow_field = self._compute_nonlinear_flowfield(
                reduced_traj[n],
                inp[n],
                static_states_n=static_states_n,
                stim_input_n=stim_input[n],
                W=W,
            )
            flow_field_list.append(flow_field)

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor,
        inp: torch.Tensor,
        delta_inp: torch.Tensor,
        **kwargs,
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

        # Unload kwargs
        delta_h_static = (
            kwargs["delta_h_static"] if "delta_h_static" in kwargs else None
        )

        traj_to_reduce = (
            kwargs["traj_to_reduce"] if "traj_to_reduce" in kwargs else states
        )

        # reshape to nxd
        states, inp, delta_inp = self._nxd(states), self._nxd(inp), self._nxd(delta_inp)

        assert inp.shape[0] == delta_inp.shape[0]
        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        # Lists for x and y velocities
        flow_field_list = []

        """
        WARNING: states is now pre-activation of network and not the activations
        this is so I can apply the activation function (some of which dont have inverses)
        to get correctly matching xs and hs 

        Previously xs and hs were assumed to be the same when gathering flow fields, but 
        this is inaccurate since the network should only be in that regime in the linear 
        case or the positive relu case.

        Here I apply the activation on states, need to make this clear in documentation
        """

        # Activity specific to regions in region list for later computations
        region_to_reduce_tmp = self.rnn.get_region_activity(
            traj_to_reduce, *self.region_list
        )
        region_tmp = self.rnn.get_region_activity(states, *self.region_list)

        # Reduce the regional trajectories and return pca object
        self._fit_traj(region_to_reduce_tmp)
        reduced_traj = self._reduce_traj(region_tmp)

        # zero out static perturbations if regions are cancelled
        if self.cancel_other_regions and delta_h_static is not None:
            delta_h_static = delta_h_static * torch.zeros_like(delta_h_static)

        for n in range(n_states):
            delta_h_static_n = delta_h_static[n] if delta_h_static is not None else None
            flow_field = self._compute_linear_flowfield(
                states[n],
                reduced_traj[n],
                inp[n],
                delta_inp[n],
                delta_h_static=delta_h_static_n,
            )
            # Reshape data back to grid
            flow_field_list.append(flow_field)

        return flow_field_list

    def _compute_nonlinear_flowfield(
        self,
        reduced_traj_n: torch.Tensor,
        inp_n: torch.Tensor,
        **kwargs,
    ) -> FlowField:
        """
        Compute a singular flow field of a nonlinear mRNN

        Args:
            reduced_traj_n (Tensor): the reduced state (n_components = 2)
            inp_n (Tensor): the input to the network corresponding to the state

        Kwargs:
            static_states_n (Tensor): activity for states not included in region list \
            defaults to a zero tensor which will assume these regions are silent

            stim_input_n (Tensor): stimulus input to network that is not weighted \
            defaults to a zero tensor for no additional network stimulus

            W (Tensor): replace mRNN weight matrix with W\
            defaults to None

        Returns:
            flow_field (FlowField): a flow field object for current inputs and network
        """
        # Unload kwargs
        static_states_n = (
            kwargs["static_states_n"]
            if "static_states_n" in kwargs
            else None  # use zero states as a dummy initialization
        )
        stim_input_n = (
            kwargs["stim_input_n"]
            if "stim_input_n" in kwargs
            else self.zero_states.clone()
        )
        W = kwargs["W"] if "W" in kwargs else None

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

        return FlowField(x_vel, y_vel, low_dim_grid, speed)

    def _compute_linear_flowfield(
        self,
        states_n: torch.Tensor,
        reduced_traj_n: torch.Tensor,
        inp_n: torch.Tensor,
        delta_inp_n: torch.Tensor,
        **kwargs,
    ) -> FlowField:
        """
        Compute a singular flow field of an affine function

        Args:
            states_n (Tensor): a particular state of the RNN
            reduced_traj_n (Tensor): the reduced state (n_components = 2)
            inp_n (Tensor): the input to the network corresponding to the state
            delta_inp_n (Tensor): perturbation of the input for state
            delta_h_static (Tensor): perturbations of state for regions treated as input

        Kwargs:
            delta_h_static (Tensor): perturbations of hidden states that are treated as input \
            (i.e. not included in region_list) \
            Defaults to a zero tensor

        Returns:
            flow_field (FlowField): a flow field object for current inputs and network
        """

        # Unload kwargs
        delta_h_static = (
            kwargs["delta_h_static"]
            if "delta_h_static" in kwargs
            else None  # dummy for initialization
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
        region_states_n = self.rnn.get_region_activity(states_n, *self.region_list)
        delta_h = inverse_grid - region_states_n

        h_states_n = self.rnn.activation(states_n)

        with torch.no_grad():
            h_next = self.linearization(
                inp_n,
                states_n,
                h_states_n,
                delta_inp_n,
                delta_h,
                delta_h_static=delta_h_static,
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

        return FlowField(x_vel, y_vel, low_dim_grid, speed)
