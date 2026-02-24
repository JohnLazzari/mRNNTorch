import torch
import numpy as np
from mrnntorch.analysis.linear import mLinearization
from dsatorch.flow_fields.flow_field import FlowField
from dsatorch.flow_fields.flow_field_finder import FlowFieldFinder
from mrnntorch.mRNN import mRNN


class mFlowFieldFinder(FlowFieldFinder[mRNN]):
    _default_hps = {
        "num_components": 2,
        "num_points": 50,
        "x_offset": 1,
        "y_offset": 1,
        "cancel_other_regions": False,
        "follow_traj": False,
        "name": "run",
        "dtype": torch.float32,
    }

    def __init__(
        self,
        rnn: mRNN,
        num_points: int = _default_hps["num_points"],
        x_offset: int = _default_hps["x_offset"],
        y_offset: int = _default_hps["y_offset"],
        cancel_other_regions: bool = _default_hps["cancel_other_regions"],
        follow_traj: bool = _default_hps["follow_traj"],
        dtype=_default_hps["dtype"],
    ):
        super().__init__(
            rnn,
            num_points,
            x_offset,
            y_offset,
            cancel_other_regions,
            follow_traj,
            dtype,
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
        self.linearization = mLinearization(rnn)

    def find_nonlinear_flow(
        self,
        states: torch.Tensor,
        inp: torch.Tensor,
        *args,
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

        Returns:
            list: FlowField object per sampled time.
        """

        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        stim_input = kwargs["stim_input"] if "stim_input" in kwargs else None
        W = kwargs["W"] if "W" in kwargs else None

        flow_field_list = []

        if self.rnn.batch_first:
            time_dim = 1
        else:
            time_dim = 0

        if states.dim() == 1:
            states = states.unsqueeze(0)

        if inp.dim() == 1:
            inp = inp.unsqueeze(0)

        states = torch.flatten(states, end_dim=-2)
        inp = torch.flatten(inp, end_dim=-2)

        assert states.shape[0] == inp.shape[0]
        n_states = states.shape[0]

        if stim_input is not None:
            stim_input = torch.flatten(stim_input, end_dim=-2)

        if not args:
            region_list = [region for region in self.rnn.region_dict]
        else:
            region_list = [region for region in args]

        reduced_traj = self._reduce_traj(states, *region_list)

        lower_bound_x = -self.x_offset
        upper_bound_x = self.x_offset
        lower_bound_y = -self.y_offset
        upper_bound_y = self.y_offset

        max_x_vels, max_y_vels, max_speeds = [], [], []
        min_x_vels, min_y_vels, min_speeds = [], [], []

        # Now going through trajectory
        for n in range(1, n_states):
            """
            This loop will compute a single flow field for state n 
            This FlowField object will then be added to a list 
            """

            if self.follow_traj:
                lower_bound_x = torch.round(
                    reduced_traj[n, 0] - self.x_offset, decimals=1
                ).item()
                upper_bound_x = torch.round(
                    reduced_traj[n, 0] + self.x_offset, decimals=1
                ).item()
                lower_bound_y = torch.round(
                    reduced_traj[n, 1] - self.y_offset, decimals=1
                ).item()
                upper_bound_y = torch.round(
                    reduced_traj[n, 1] + self.y_offset, decimals=1
                ).item()

            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Repeat along the batch dimension to match the grid
            full_act_batch = states[n].repeat(low_dim_grid.shape[0], 1)
            full_inp_batch = inp[n].repeat(low_dim_grid.shape[0], 1)

            x_0_flow = self._compute_full_trajectory(
                inverse_grid, full_act_batch, *region_list
            )

            with torch.no_grad():
                # Current timestep input
                # Get activity for current timestep
                if stim_input is None:
                    full_stim_batch = torch.zeros_like(x_0_flow, dtype=self.dtype)
                else:
                    full_stim_batch = stim_input[n, :].repeat(low_dim_grid.shape[0], 1)

                _, h = self.rnn(
                    full_inp_batch.unsqueeze(time_dim),
                    x_0_flow,
                    x_0_flow,
                    full_stim_batch.unsqueeze(time_dim),
                    noise=False,
                    W_rec=W,
                )

            # Get activity for regions of interest
            cur_region_h = self.rnn.get_region_activity(h, *region_list)
            cur_region_h = torch.reshape(cur_region_h, (-1, cur_region_h.shape[-1]))
            cur_region_h = self.reduce_obj.transform(cur_region_h)
            cur_region_h = torch.tensor(cur_region_h, dtype=self.dtype)

            x_vel, y_vel = self._compute_velocity(cur_region_h, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field = FlowField(x_vel, y_vel, low_dim_grid, speed)
            flow_field_list.append(flow_field)

            # append max values to lists
            max_x_vels.append(flow_field.max_x_vel)
            max_y_vels.append(flow_field.max_y_vel)
            max_speeds.append(flow_field.max_speed)

            # append min values to lists
            min_x_vels.append(flow_field.min_x_vel)
            min_y_vels.append(flow_field.min_y_vel)
            min_speeds.append(flow_field.min_speed)

        if verbose:
            mean_max_x_vel, std_max_x_vel = np.mean(max_x_vels), np.std(max_x_vels)
            mean_max_y_vel, std_max_y_vel = np.mean(max_y_vels), np.std(max_y_vels)
            mean_max_speed, std_max_speed = np.mean(max_speeds), np.std(max_speeds)

            mean_min_x_vel, std_min_x_vel = np.mean(min_x_vels), np.std(min_x_vels)
            mean_min_y_vel, std_min_y_vel = np.mean(min_x_vels), np.std(min_y_vels)
            mean_min_speed, std_min_speed = np.mean(min_x_vels), np.std(min_speeds)

            print("======================")
            print("Flow Field Statistics:")
            print(
                f"mean max x vel: {mean_max_x_vel} +/- {std_max_x_vel}   mean min x vel: {mean_min_x_vel} +/- {std_min_x_vel}"
            )
            print(
                f"mean max y vel: {mean_max_y_vel} +/- {std_max_y_vel}   mean min y vel: {mean_min_y_vel} +/- {std_min_y_vel}"
            )
            print(
                f"mean max speed: {mean_max_speed} +/- {std_max_speed}   mean min speed: {mean_min_speed} +/- {std_min_speed}"
            )
            print("======================")

        return flow_field_list

    def find_linear_flow(
        self,
        states: torch.Tensor,
        *args,
    ) -> list:
        """Compute linearized flow fields in a 2D subspace.

        Similar to :func:`flow_field`, but uses a local linear approximation (Jacobian)
        of the dynamics around points on the trajectory instead of a full forward
        step. Assumes no external input to the selected regions.

        Args:
            states (torch.Tensor): Hidden activations over time for selected regions, [n, d]
        Returns:
            list: FlowField objects per sampled time.
        """
        # Lists for x and y velocities
        flow_field_list = []

        states = torch.flatten(states, end_dim=-2)
        n_states = states.shape[0]

        # Gather region list from args, include all regions if args empty
        if not args:
            region_list = [region for region in self.rnn.region_dict]
        else:
            region_list = [region for region in args]

        # Activity specific to regions in region list for later computations
        region_act = self.rnn.get_region_activity(states, *args)
        # Reduce the regional trajectories and return pca object
        reduced_traj = self._reduce_traj(region_act, *region_list)

        # Grid offsets
        lower_bound_x = -self.x_offset
        upper_bound_x = self.x_offset
        lower_bound_y = -self.y_offset
        upper_bound_y = self.y_offset

        for n in range(1, n_states):
            # If follow trajectory is true get grid centered around current t
            # This will make a different grid for each state (n grids)
            if self.follow_traj:
                lower_bound_x = torch.round(
                    reduced_traj[n, 0] - self.x_offset, decimals=1
                ).item()
                upper_bound_x = torch.round(
                    reduced_traj[n, 0] + self.x_offset, decimals=1
                ).item()
                lower_bound_y = torch.round(
                    reduced_traj[n, 1] - self.y_offset, decimals=1
                ).item()
                upper_bound_y = torch.round(
                    reduced_traj[n, 1] + self.y_offset, decimals=1
                ).item()

            # Inverse the grid to pass through RNN
            low_dim_grid, inverse_grid = self._inverse_grid(
                lower_bound_x,
                upper_bound_x,
                lower_bound_y,
                upper_bound_y,
            )

            # Get a perturbation of the activity
            x_0_flow = inverse_grid - region_act[n, :]

            with torch.no_grad():
                # Return jacobian found from current trajectory
                jac_rec = self.linearization.jacobian(states[n, :], *region_list)
                # Get next h
                h = region_act[n, :] + (jac_rec @ x_0_flow.T).T

            # Put next h into a grid format
            cur_region_h = self.reduce_obj.transform(h)
            cur_region_h = torch.tensor(cur_region_h, dtype=self.dtype)

            # Compute velocities between gathered trajectory of grid and original grid values
            x_vel, y_vel = self._compute_velocity(cur_region_h, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            # Reshape data back to grid
            flow_field_list.append(FlowField(x_vel, y_vel, low_dim_grid, speed))

        return flow_field_list

    def _reduce_traj(self, trajectory: torch.Tensor, *args) -> torch.Tensor:
        """
        Fit PCA object and transform trajectory

        Args:
            trajectory (Tensor): states to reduce
            args (str, ...): regions to gather

        Returns:
            Tensor: reduced states
        """
        # Gather activity for specified region and cell type
        temp_act = self.rnn.get_region_activity(trajectory, *args)
        temp_act = torch.reshape(temp_act, (-1, temp_act.shape[-1]))

        # Do PCA on the specified region(s)
        self.reduce_obj.fit(temp_act)
        reduced_traj = self.reduce_obj.transform(temp_act)
        reduced_traj = torch.from_numpy(reduced_traj)

        return reduced_traj

    def _compute_full_trajectory(
        self, grid: torch.Tensor, full_act_batch: torch.Tensor, *args
    ) -> torch.Tensor:
        """
        Create a grid with containing all regions

        Args:
           grid (Tensor): The inverse grid containing states only for selected regions
           [n, num_units_in_selected_regions]

           full_act_batch (Tensor): The full activity state of the network (containing)
           selected and non-selected regions) at current chosen state, repeated over
           batch dim to match the number of points in the grid.
           [n, num_units_in_network]
        """
        # Gather batches of grids with trial activity at each timestep
        grid_region_idx = 0
        x_0_flow = []
        for region in self.rnn.region_dict:
            if region in args:
                x_0_flow.append(
                    grid[
                        :,
                        grid_region_idx : grid_region_idx
                        + self.rnn.region_dict[region].num_units,
                    ]
                )
                grid_region_idx += self.rnn.region_dict[region].num_units
            else:
                # Get activity for non-specified regions (either from cache or compute)
                if self.cancel_other_regions:
                    region_activity = torch.zeros_like(
                        self.rnn.get_region_activity(full_act_batch, region)
                    )
                else:
                    region_activity = self.rnn.get_region_activity(
                        full_act_batch, region
                    )
                x_0_flow.append(region_activity)
        x_0_flow = torch.cat(x_0_flow, dim=-1)
        return x_0_flow
