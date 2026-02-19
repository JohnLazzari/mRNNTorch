import torch
import numpy as np
from sklearn.decomposition import PCA
from numpy.typing import NDArray
from mrnntorch.analysis.linear.linearization import Linearization
from mrnntorch.analysis.flow_fields.flow_field import FlowField
from mrnntorch.mRNN import mRNN


class FlowFieldFinder:
    _default_hps = {
        "num_components": 2,
        "num_points": 50,
        "x_offset": 1,
        "y_offset": 1,
        "cancel_other_regions": False,
        "follow_traj": False,
        "name": "run",
        "reduction_method": "pca",
        "dtype": torch.float32,
    }

    def __init__(
        self,
        mrnn: mRNN,
        num_points: int = _default_hps["num_points"],
        x_offset: int = _default_hps["x_offset"],
        y_offset: int = _default_hps["y_offset"],
        cancel_other_regions: bool = _default_hps["cancel_other_regions"],
        follow_traj: bool = _default_hps["follow_traj"],
        dtype: float = _default_hps["dtype"],
    ):
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
        # Hyperparameters
        self.mrnn = mrnn
        self.num_points = num_points
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.cancel_other_regions = cancel_other_regions
        self.follow_traj = follow_traj
        self.dtype = dtype

        # class objects
        self.reduce_obj = PCA(n_components=2)
        self.linearization = Linearization(self.mrnn)

    def find_nonlinear_flow(
        self,
        states: torch.Tensor,
        inp: torch.Tensor,
        *args,
        stim_input: torch.Tensor = None,
        W: torch.Tensor = None,
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

        flow_field_list = []

        if self.mrnn.batch_first:
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
            region_list = [region for region in self.mrnn.region_dict]
        else:
            region_list = [region for region in args]

        reduced_traj = self._reduce_traj(states, *region_list)

        lower_bound_x = -self.x_offset
        upper_bound_x = self.x_offset
        lower_bound_y = -self.y_offset
        upper_bound_y = self.y_offset

        # Now going through trajectory
        for n in range(1, n_states):
            if self.follow_traj:
                lower_bound_x = np.round(reduced_traj[n, 0] - self.x_offset, decimals=1)
                upper_bound_x = np.round(reduced_traj[n, 0] + self.x_offset, decimals=1)
                lower_bound_y = np.round(reduced_traj[n, 1] - self.y_offset, decimals=1)
                upper_bound_y = np.round(reduced_traj[n, 1] + self.y_offset, decimals=1)

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

                _, h = self.mrnn(
                    full_inp_batch.unsqueeze(time_dim),
                    x_0_flow,
                    x_0_flow,
                    full_stim_batch.unsqueeze(time_dim),
                    noise=False,
                    W_rec=W,
                )

            # Get activity for regions of interest
            cur_region_h = self.mrnn.get_region_activity(h, *region_list)
            cur_region_h = torch.reshape(cur_region_h, (-1, cur_region_h.shape[-1]))
            cur_region_h = self.reduce_obj.transform(cur_region_h)
            cur_region_h = torch.tensor(cur_region_h, dtype=self.dtype)

            x_vel, y_vel = self._compute_velocity(cur_region_h, low_dim_grid)
            speed = self._compute_speed(x_vel, y_vel)

            # Reshape to match FlowField object requirements
            x_vel, y_vel, low_dim_grid, speed = self._reshape_vals(
                x_vel, y_vel, low_dim_grid, speed
            )

            flow_field_list.append(FlowField(x_vel, y_vel, low_dim_grid, speed))

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
            region_list = [region for region in self.mrnn.region_dict]
        else:
            region_list = [region for region in args]

        # Activity specific to regions in region list for later computations
        region_act = self.mrnn.get_region_activity(states, *args)
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
                lower_bound_x = np.round(reduced_traj[n, 0] - self.x_offset, decimals=1)
                upper_bound_x = np.round(reduced_traj[n, 0] + self.x_offset, decimals=1)
                lower_bound_y = np.round(reduced_traj[n, 1] - self.y_offset, decimals=1)
                upper_bound_y = np.round(reduced_traj[n, 1] + self.y_offset, decimals=1)

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
                jac_rec = self.linearization.jacobian(
                    states[n, :], *region_list, alpha=1
                )
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

    def _reduce_traj(self, trajectory: torch.Tensor, *args) -> NDArray:
        """
        Fit PCA object and transform trajectory

        Args:
            trajectory (Tensor): states to reduce
            args (str, ...): regions to gather

        Returns:
            Tensor: reduced states
        """
        # Gather activity for specified region and cell type
        temp_act = self.mrnn.get_region_activity(trajectory, *args)
        temp_act = torch.reshape(temp_act, (-1, temp_act.shape[-1]))

        # Do PCA on the specified region(s)
        self.reduce_obj.fit(temp_act)
        reduced_traj = self.reduce_obj.transform(temp_act)

        return reduced_traj

    def _inverse_grid(
        self,
        lower_bound_x: float,
        upper_bound_x: float,
        lower_bound_y: float,
        upper_bound_y: float,
        expand_dims: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Obtain a low dimensional grid and its projection to higher dim state
        space

        Args:
            lower_bound_x (float): lower bound of grid in x direction
            upper_bound_x (float): upper bound of grid in x direction
            lower_bound_y (float): lower bound of grid in y direction
            upper_bound_y (float): upper bound of grid in y direction

        Returns:
            tuple: the low dimensional and projected low dimensional grid
        """
        # Num points is along each axis, not in total
        x = np.linspace(lower_bound_x, upper_bound_x, self.num_points)
        y = np.linspace(lower_bound_y, upper_bound_y, self.num_points)

        # Gather 2D grid for flow fields
        xv, yv = np.meshgrid(x, y)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)

        # Convert the grid to a tensor and flatten for PCA
        low_dim_grid = np.concatenate((xv, yv), axis=-1)
        low_dim_grid = torch.tensor(low_dim_grid, dtype=self.dtype)
        low_dim_grid = torch.flatten(low_dim_grid, start_dim=0, end_dim=1)

        # Inverse PCA to input grid into network
        inverse_grid = self.reduce_obj.inverse_transform(low_dim_grid)
        inverse_grid = torch.tensor(inverse_grid, dtype=self.dtype)

        if expand_dims:
            low_dim_grid = torch.reshape(
                low_dim_grid, (self.num_points, self.num_points, 2)
            )
            inverse_grid = torch.reshape(
                inverse_grid, (self.num_points, self.num_points, inverse_grid.shape[-1])
            )

        return low_dim_grid, inverse_grid

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
        for region in self.mrnn.region_dict:
            if region in args:
                x_0_flow.append(
                    grid[
                        :,
                        grid_region_idx : grid_region_idx
                        + self.mrnn.region_dict[region].num_units,
                    ]
                )
                grid_region_idx += self.mrnn.region_dict[region].num_units
            else:
                # Get activity for non-specified regions (either from cache or compute)
                if self.cancel_other_regions:
                    region_activity = torch.zeros_like(
                        self.mrnn.get_region_activity(full_act_batch, region)
                    )
                else:
                    region_activity = self.mrnn.get_region_activity(
                        full_act_batch, region
                    )
                x_0_flow.append(region_activity)
        x_0_flow = torch.cat(x_0_flow, dim=-1)
        return x_0_flow

    def _compute_velocity(
        self, h_next: torch.Tensor, h_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """compute velocity, or h_next - h_prev"""
        x_vel = h_next[..., 0] - h_prev[..., 0]
        y_vel = h_next[..., 1] - h_prev[..., 1]
        return x_vel, y_vel

    def _compute_speed(self, x_vel: torch.Tensor, y_vel: torch.Tensor) -> torch.Tensor:
        """compute magnitude of velocities"""
        speed = torch.sqrt(x_vel**2 + y_vel**2)
        return speed / speed.max()

    def _reshape_vals(
        self,
        x_vels: torch.Tensor,
        y_vels: torch.Tensor,
        grid: torch.Tensor,
        speeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Broadcast data to FlowField format

        Args:
            x_vels (Tensor): x velocities
            y_vels (Tensor): y velocities
            grid (Tensor): low dimensional grid coordinates
            speeds (Tensor): magnitude of x_vels and y_vels
        """
        # Reshape to match FlowField object requirements
        x_vels = torch.reshape(x_vels, (self.num_points, self.num_points))
        y_vels = torch.reshape(y_vels, (self.num_points, self.num_points))
        grid = torch.reshape(grid, (self.num_points, self.num_points, 2))
        speeds = torch.reshape(speeds, (self.num_points, self.num_points))
        return x_vels, y_vels, grid, speeds
