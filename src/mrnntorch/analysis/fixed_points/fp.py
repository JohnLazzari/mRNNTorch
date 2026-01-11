import torch
import numpy as np
import pickle
from typing import Self


class FixedPointCollection:
    """
    A class for storing fixed points and associated data.
    """

    """ List of class attributes that represent data corresponding to fixed
    points. All of these refer to Numpy arrays with axis 0 as the batch
    dimension. Thus, each is concatenatable using np.concatenate(..., axis=0).
    """
    _data_attrs_fp = [
        "xstar",
        "x_init",
        "inputs",
        "F_xstar",
        "qstar",
        "dq",
        "n_iters",
    ]

    """ List of class attributes that apply to all fixed points
    (i.e., these are not indexed per fixed point). """
    _nonspecific_attrs = [
        "dtype",
        "dtype_complex",
        "tol_unique",
        "verbose",
    ]

    def __init__(
        self,
        xstar: torch.Tensor,  # Fixed-point specific data
        x_init: torch.Tensor = None,
        inputs: torch.Tensor = None,
        F_xstar: torch.Tensor = None,
        qstar: torch.Tensor = None,
        dq: torch.Tensor = None,
        n_iters: torch.Tensor = None,
        tol_unique: float = 1e-3,
        dtype: torch.float = torch.float32,
        dtype_complex: torch.complex = torch.complex64,
        verbose: bool = False,
    ):
        """
        Initializes a FixedPoints object with all input arguments as class
        properties.

        Optional args:

            xstar: [n x n_states] tensor with row xstar[i, :]
            specifying an the fixed point identified from x_init[i, :].
            Default: None.

            x_init: [n x n_states] tensor with row x_init[i, :]
            specifying the initial state from which xstar[i, :] was optimized.
            Default: None.

            inputs: [n x n_inputs] tensor with row inputs[i, :]
            specifying the input to the RNN during the optimization of
            xstar[i, :]. Default: None.

            F_xstar: [n x n_states] tensor with F_xstar[i, :]
            specifying RNN state after transitioning from the fixed point in
            xstar[i, :]. If the optimization succeeded (e.g., to 'tol') and
            identified a stable fixed point, the state should not move
            substantially from the fixed point (i.e., xstar[i, :] should be
            very close to F_xstar[i, :]). Default: None.

            qstar: [n,] tensor with qstar[i] containing the
            optimized objective (1/2)(x-F(x))^T(x-F(x)), where
            x = xstar[i, :]^T and F is the RNN transition function (with the
            specified constant inputs). Default: None.

            dq: [n,] tensor with dq[i] containing the absolute
            difference in the objective function after (i.e., qstar[i]) vs
            before the final gradient descent step of the optimization of
            xstar[i, :]. Default: None.

            n_iters: [n,] tensor with n_iters[i] as the number of
            gradient descent iterations completed to yield xstar[i, :].
            Default: None.

            tol_unique: Positive scalar specifying the numerical precision
            required to label two fixed points as being unique from one
            another. Two fixed points are considered unique if the 2-norm of
            the difference between their concatenated (xstar, inputs) is
            greater than this tolerance. Default: 1e-3.

            dtype: Data type for representing all of the object's data.
            Default: torch.float32.

            dtype_complex: Data type for representing complex values.
            Default: torch.complex32.

            verbose: Bool indicating whether to print status updates.

        """

        # These apply to all fixed points
        # (one value each, rather than one value per fixed point).
        self.tol_unique = tol_unique
        self.dtype = dtype
        self.dtype_complex = dtype_complex
        self.verbose = verbose

        self.n, self.n_states = xstar.shape

        if inputs is not None:
            self.n_inputs = inputs.shape[1]
        else:
            self.n_inputs = None

        # Common FP attributes
        self.xstar = xstar
        self.x_init = x_init
        self.inputs = inputs
        self.F_xstar = F_xstar
        self.qstar = qstar
        self.dq = dq
        self.n_iters = n_iters

        self.assert_valid_shapes()

    def __setitem__(self, index: int | list | slice, fps: Self):
        """Indexes into a subset of the fixed points and their associated data.

        Usage:
            fps_subset = fps[index]

        Args:
            index: a slice object for indexing into the FixedPoints data.

        Returns:
            A FixedPoints object containing a subset of the data from the
            current FixedPoints object, as specified by index.
        """
        assert isinstance(fps, type(self)), (
            "fps must be a self object but was %s." % type(fps)
        )
        if isinstance(index, tuple):
            assert len(index) <= 1  # only 1d slicing allowed
        # Preserve shape of objects if index is an int
        if isinstance(index, int):
            index = slice(index, index + 1)

        for attr_name in self._data_attrs_fp:
            attr_val = getattr(self, attr_name)
            if attr_val is not None:
                attr_val[index] = getattr(fps, attr_name)

    def __getitem__(self, index: int | list | slice) -> Self:
        """Indexes into a subset of the fixed points and their associated data.

        Usage:
            fps_subset = fps[index]

        Args:
            index: a slice object for indexing into the FixedPoints data.

        Returns:
            A FixedPoints object containing a subset of the data from the
            current FixedPoints object, as specified by index.
        """
        if isinstance(index, tuple):
            assert len(index) <= 1  # only 1d slicing allowed
        # Preserve shape of objects if index is an int
        if isinstance(index, int):
            index = slice(index, index + 1)

        kwargs = self._nonspecific_kwargs
        for attr_name in self._data_attrs_fp:
            attr_val = getattr(self, attr_name)
            indexed_val = self._safe_index(attr_val, index)
            kwargs[attr_name] = indexed_val
        return type(self)(**kwargs)

    def __len__(self) -> int:
        """Returns the number of fixed points stored in the object."""
        return self.n

    def __contains__(self, fp: Self) -> bool:
        """Checks whether a specified fixed point is contained in the object.

        Args:
            fp: A FP object containing exactly one fixed point.

        Returns:
            bool indicating whether any fixed point matches fp.
        """

        idx = self.find(fp)

        return idx.numel() > 0

    def get_unique(self) -> Self:
        """
        Identifies unique fixed points. Among duplicates identified,
        this keeps the one with smallest qstar.

        Args:
            None.

        Returns:
            A FixedPoints object containing only the unique fixed points and
            their associated data. Uniqueness is determined down to tol_unique.
        """
        assert self.xstar is not None, (
            "Cannot find unique fixed points because self.xstar is None."
        )

        idx_keep = []
        idx_checked = torch.zeros(size=(self.n,), dtype=bool)
        for idx in range(self.n):
            if idx_checked[idx]:
                # If this FP matched others, we've already determined which
                # of those matching FPs to keep. Repeating would simply
                # identify the same FP to keep.
                continue

            # Don't compare against FPs we've already checked
            idx_check = torch.where(~idx_checked)[0]
            fps_check = self[idx_check]  # only check against these FPs
            idx_idx_check = fps_check.find(self[idx])  # indexes into fps_check
            idx_match = idx_check[idx_idx_check]  # indexes into self

            if len(idx_match) == 1:
                # Only matches with itself
                idx_keep.append(idx)
            else:
                qstars_match = self._safe_index(self.qstar, idx_match)
                idx_candidate = idx_match[torch.argmin(qstars_match)]
                idx_keep.append(int(idx_candidate))
                idx_checked[idx_match] = True

        return self[idx_keep]

    def transform(self, U: torch.Tensor, offset: float = 0.0) -> Self:
        """Apply an affine transformation to the state-space representation.
        This may be helpful for plotting fixed points in a given linear
        subspace (e.g., PCA or an RNN readout space).

        Args:
            U: shape (n_states, k) numpy array projection matrix.

            offset (optional): shape (k,) numpy translation vector. Default: 0.

        Returns:
            A FixedPoints object.
        """
        kwargs = self.kwargs

        # These are all transformed. All others are not.
        for attr_name in ["xstar", "x_init", "F_xstar"]:
            kwargs[attr_name] = getattr(self, attr_name) @ U + offset

        return type(self)(**kwargs)

    def find(self, fp: Self) -> torch.Tensor:
        """Searches in the current FixedPoints object for matches to a
        specified fixed point. Two fixed points are defined as matching
        if the 2-norm of the difference between their concatenated (xstar,
        inputs) is within tol_unique).

        Args:
            fp: A FixedPoints object containing exactly one fixed point.

        Returns:
            shape (n_matches,) tensor specifying indices into the current
            FixedPoints object where matches to fp were found.
        """

        # If not found or comparison is impossible (due to type or shape),
        # follow convention of torch.where and return an empty tensor.
        result = torch.empty(0, dtype=int)

        if isinstance(fp, FixedPointCollection):
            if fp.n_states == self.n_states and fp.n_inputs == self.n_inputs:
                if self.inputs is None:
                    self_data_nxd = self.xstar
                    arg_data_nxd = fp.xstar
                else:
                    self_data_nxd = torch.cat((self.xstar, self.inputs), axis=1)
                    arg_data_nxd = torch.cat((fp.xstar, fp.inputs), axis=1)

                norm_diffs_n = torch.linalg.norm(self_data_nxd - arg_data_nxd, axis=1)

                result = torch.where(norm_diffs_n <= self.tol_unique)[0]

        return result

    def update(self, new_fps: Self):
        """Combines the entries from another FixedPoints object into this
        object.

        Args:
            new_fps: a FixedPoints object containing the entries to be
            incorporated into this FixedPoints object.

        Returns:
            None

        Raises:
            AssertionError if the non-fixed-point specific attributes of
            new_fps do not match those of this FixedPoints object.

            AssertionError if any data attributes are found in one but not both
            FixedPoints objects (especially relevant for decomposed Jacobians).

            AssertionError if the updated object has inconsistent data shapes.
        """

        self._assert_matching_nonspecific_attrs([new_fps])

        for attr_name in self._data_attrs_fp:
            this_has = hasattr(self, attr_name)
            that_has = hasattr(new_fps, attr_name)

            self_attr = getattr(self, attr_name)
            other_attr = getattr(new_fps, attr_name)

            assert this_has == that_has, (
                "One but not both FixedPoints objects have %s. "
                "FixedPoints.update does not currently support this "
                "configuration." % attr_name
            )

            # Ensure both objects have the attribute
            if this_has and that_has:
                # Now check whether, one, both, or neither attrs are None
                if self_attr is None and other_attr is None:
                    setattr(self, attr_name, None)
                elif self_attr is None and other_attr is not None:
                    setattr(self, attr_name, other_attr)
                elif self_attr is not None and other_attr is None:
                    setattr(self, attr_name, self_attr)
                else:
                    cat_attr = torch.cat((self_attr, other_attr), axis=0)
                    setattr(self, attr_name, cat_attr)

        self.n = self.n + new_fps.n
        self.assert_valid_shapes()

    def save(self, save_path: str):
        """Saves all data contained in the FixedPoints object.

        Args:
            save_path: A string containing the path at which to save
            (including directory, filename, and arbitrary extension).

        Returns:
            None.
        """
        if self.verbose:
            print("Saving FixedPoints object.")

        self.assert_valid_shapes()

        file = open(save_path, "wb")
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def restore(self, restore_path: str):
        """Restores data from a previously saved FixedPoints object.

        Args:
            restore_path: A string containing the path at which to find a
            previously saved FixedPoints object (including directory, filename,
            and extension).

        Returns:
            None.
        """
        if self.verbose:
            print("Restoring FixedPoints object.")
        file = open(restore_path, "rb")
        restore_data = file.read()
        file.close()
        self.__dict__ = pickle.loads(restore_data)
        self.assert_valid_shapes()

    def assert_valid_shapes(self):
        """Checks that all data attributes reflect the same number of fixed
        points.

        Raises:
            AssertionError if any non-None data attribute does not have
            .shape[0] as self.n.
        """
        n = self.n
        for attr_name in FixedPointCollection._data_attrs_fp:
            data = getattr(self, attr_name)
            if data is not None:
                assert data.shape[0] == self.n, (
                    "Detected %d fixed points, but %s.shape is %s "
                    "(shape[0] should be %d" % (n, attr_name, str(data.shape), n)
                )

    @staticmethod
    def concatenate(fps_seq: list):
        """Join a sequence of FixedPoints objects.

        Args:
            fps_seq: sequence of FixedPoints objects. All FixedPoints objects
            must have the following attributes in common:
                n_states
                n_inputs

        Returns:
            A FixedPoints objects containing the concatenated FixedPoints data.
        """

        assert len(fps_seq) > 0, "Cannot concatenate empty list."
        FixedPointCollection._assert_matching_nonspecific_attrs(fps_seq)

        kwargs = {}

        for attr_name in FixedPointCollection._nonspecific_attrs:
            kwargs[attr_name] = getattr(fps_seq[0], attr_name)

        for attr_name in FixedPointCollection._data_attrs_fp:
            if all((hasattr(fps, attr_name) for fps in fps_seq)):
                cat_list = [getattr(fps, attr_name) for fps in fps_seq]

                if all([attr is None for attr in cat_list]):
                    cat_attr = None
                elif any([attr is None for attr in cat_list]):
                    cat_attr = None
                else:
                    cat_attr = torch.cat(cat_list)

                kwargs[attr_name] = cat_attr

        return FixedPointCollection(**kwargs)

    @property
    def is_single_fixed_point(self) -> bool:
        return self.n == 1

    @property
    def kwargs(self) -> dict:
        """Returns dict of keyword arguments necessary for reinstantiating a
        (shallow) copy of this FixedPoints object, i.e.,

        fp_copy  = FixedPoints(**fp.kwargs)
        """

        kwargs = self._nonspecific_kwargs

        for attr_name in self._data_attrs_fp:
            kwargs[attr_name] = getattr(self, attr_name)

        return kwargs

    @staticmethod
    def _assert_matching_nonspecific_attrs(fps_seq: list):
        for attr_name in FixedPointCollection._nonspecific_attrs:
            items = [getattr(fps, attr_name) for fps in fps_seq]
            for item in items:
                assert item == items[0], (
                    "Cannot concatenate FixedPoints because of mismatched %s "
                    "(%s is not %s)" % (attr_name, str(items[0]), str(item))
                )

    @staticmethod
    def _safe_index(x: torch.Tensor, idx: int | list | slice) -> torch.Tensor:
        """Safe method for indexing into a tensor that might be None.

        Args:
            x: Either None or a tensor.
            idx: Positive int or index-compatible argument for indexing into x.
        """
        if x is None:
            return None
        else:
            return x[idx]

    @property
    def _nonspecific_kwargs(self) -> dict:
        # These are not specific to individual fixed points.
        # Thus, simple copy, no indexing required
        return {"dtype": self.dtype, "tol_unique": self.tol_unique}
