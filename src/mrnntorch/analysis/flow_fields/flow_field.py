import torch
from typing import Self


class FlowField:
    _definable_attr = ["x_vels", "y_vels", "grid", "speeds"]

    def __init__(
        self,
        x_vels: torch.Tensor,
        y_vels: torch.Tensor,
        grid: torch.Tensor,
        speeds: torch.Tensor,
    ):
        """
        Flow field object that stores grids, velocities, speeds, and allows for indexing

        Args:
            x_vels (tensor): [n, n] x velocities
            y_vels (tensor): [n, n] y velocities
            grid (tensor): [n, n, d] grid coordinates
            speeds (tensor): [n, n] speeds
            grid_rows (int): number of rows in grid
            grid_columns (int): number of columns in grid
            state_size (int): dimensions of state in grid
        """
        # Flow storage
        self.x_vels = x_vels
        self.y_vels = y_vels
        self.speeds = speeds
        self.grid = grid

        # Assert appropriate shapes
        assert self.speeds.dim() == 2
        assert self.x_vels.shape == self.y_vels.shape == self.speeds.shape

        assert self.grid.dim() == 3

        self.grid_rows = self.grid.shape[0]
        self.grid_columns = self.grid.shape[1]
        self.state_size = self.grid.shape[2]

    def __getitem__(
        self,
        idx: int
        | slice
        | tuple[int, slice]
        | tuple[slice, int]
        | tuple[int, int]
        | tuple[slice, slice],
    ) -> Self:
        """
        Returns a FlowField object at specified index
        Only _definable_attr are changed
        Allows for 1d or 2d indexing or slicing with appropriate broadcasting

        Args:
            idx: integer, slice, or tuple of either integers or slices
        """
        # ensure broadcasting is handled
        norm_idx = self._normalize_index(idx)
        kwargs = {}
        for attr_name in self._definable_attr:
            attr_val = getattr(self, attr_name)
            idx_attr_val = attr_val[norm_idx]
            """
            four possible ways to index: 
                flow[n], flow[:, n], flow[n, :], flow[n, n]
            """
            kwargs[attr_name] = idx_attr_val
        return type(self)(**kwargs)

    def __len__(self) -> int:
        """Total number of grid coordinates"""
        return self.grid_rows * self.grid_columns

    def _normalize_index(
        self,
        idx: int
        | slice
        | tuple[int, slice]
        | tuple[slice, int]
        | tuple[int, int]
        | tuple[slice, slice],
    ) -> tuple:
        """
        Helper function to ensure appropriate broadcasting

        Args:
            idx: int, slice, or 2d tuple of either

        Returns:
            tuple: 2d tuple ensuring appropriate broadcasting after indexing
        """
        # If a single int or slice is passed, convert to tuple of length 2
        if isinstance(idx, int):
            ext_idx = (slice(idx, idx + 1), slice(None))
        elif isinstance(idx, slice):
            ext_idx = (idx, slice(None))
        elif isinstance(idx, tuple):
            # Only allow 2D indexing
            assert len(idx) <= 2, "Can only index max 2D"
            # Fill missing dimensions with slice(None)
            if len(idx) == 1:
                idx = (idx[0], slice(None))
            # Convert any ints in the tuple to slices/lists to keep dims
            ext_idx = tuple(slice(i, i + 1) if isinstance(i, int) else i for i in idx)
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")
        return ext_idx
