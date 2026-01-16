import torch
import numpy as np
import time
from copy import deepcopy

from .fp import FixedPointCollection
from mrnntorch.analysis.linear.linearization import Linearization
from mrnntorch.mRNN import mRNN


class FixedPointFinder:
    _default_hps = {
        "lr_init": 1e-4,
        "lr_patience": 5,
        "lr_factor": 0.95,
        "lr_cooldown": 0,
        "tol_q": 1e-12,
        "tol_dq": 1e-20,
        "max_iters": 5000,
        "do_rerun_q_outliers": False,
        "outlier_q_scale": 10.0,
        "do_exclude_distance_outliers": True,
        "outlier_distance_scale": 10.0,
        "tol_unique": 1e-3,
        "max_n_unique": np.inf,
        "dtype": "float32",
        "random_seed": 0,
        "verbose": True,
        "super_verbose": False,
        "n_iters_per_print_update": 100,
        "batch_first": True,
    }

    @classmethod
    def default_hps(cls):
        """Returns a deep copy of the default hyperparameters dict.

        The deep copy protects against external updates to the defaults, which
        in turn protects against unintended interactions with the hashing done
        by the Hyperparameters class.

        Args:
            None.

        Returns:
            dict of hyperparameters.


        """
        return deepcopy(cls._default_hps)

    def __init__(
        self,
        mrnn: mRNN,
        lr_init: float = _default_hps["lr_init"],
        lr_patience: float = _default_hps["lr_patience"],
        lr_factor: float = _default_hps["lr_factor"],
        lr_cooldown: float = _default_hps["lr_cooldown"],
        tol_q: float = _default_hps["tol_q"],
        tol_dq: float = _default_hps["tol_dq"],
        max_iters: int = _default_hps["max_iters"],
        do_rerun_q_outliers: bool = _default_hps["do_rerun_q_outliers"],
        outlier_q_scale: float = _default_hps["outlier_q_scale"],
        do_exclude_distance_outliers: bool = _default_hps[
            "do_exclude_distance_outliers"
        ],
        outlier_distance_scale: float = _default_hps["outlier_distance_scale"],
        tol_unique: float = _default_hps["tol_unique"],
        max_n_unique: int = _default_hps["max_n_unique"],
        dtype: str = _default_hps["dtype"],
        random_seed: int = _default_hps["random_seed"],
        verbose: bool = _default_hps["verbose"],
        super_verbose: bool = _default_hps["super_verbose"],
        n_iters_per_print_update: int = _default_hps["n_iters_per_print_update"],
    ):
        """Creates a FixedPointFinder object.

        Optimization terminates once every initialization satisfies one or
        both of the following criteria:
            1. q < tol_q
            2. dq < tol_dq * learning_rate

        Args:
            rnn_cell: A Pytorch RNN

            tol_q (optional): A positive scalar specifying the optimization
            termination criteria on each q-value. Default: 1e-12.

            tol_dq (optional): A positive scalar specifying the optimization
            termination criteria on the improvement of each q-value (i.e.,
            "dq") from one optimization iteration to the next. Default: 1e-20.

            max_iters (optional): A non-negative integer specifying the
            maximum number of gradient descent iterations allowed.
            Optimization terminates upon reaching this iteration count, even
            if 'tol' has not been reached. Default: 5000.

            do_rerun_q_outliers (optional): A bool indicating whether or not
            to run additional optimization iterations on putative outlier
            states, identified as states with large q values relative to the
            median q value across all identified fixed points (i.e., after
            the initial optimization ran to termination). These additional
            optimizations are run sequentially (even if method is 'joint').
            Default: False.

            outlier_q_scale (optional): A positive float specifying the q
            value for putative outlier fixed points, relative to the median q
            value across all identified fixed points. Default: 10.

            do_exclude_distance_outliers (optional): A bool indicating
            whether or not to discard states that are far away from the set
            of initial states, as measured by a normalized euclidean
            distance. If true, states are evaluated and possibly discarded
            after the initial optimization runs to termination.
            Default: True.

            outlier_distance_scale (optional): A positive float specifying a
            normalized distance cutoff used to exclude distance outliers. All
            distances are calculated relative to the centroid of the
            initial_states and are normalized by the average distance-to-
            centroid of the initial_states. Default: 10.

            tol_unique (optional): A positive scalar specifying the numerical
            precision required to label two fixed points as being unique from
            one another. Two fixed points will be considered unique if they
            differ by this amount (or more) along any dimension. This
            tolerance is used to discard numerically similar fixed points.
            Default: 1e-3.

            max_n_unique (optional): A positive integer indicating the max
            number of unique fixed points to keep. If the number of unique
            fixed points identified exceeds this value, points are randomly
            dropped. Default: np.inf.

            dtype: string indicating the data type to use for all numerical ops
            and objects. Default: 'float32'

            random_seed: Seed for numpy random number generator. Default: 0.

            verbose (optional): A bool indicating whether to print high-level
            status updates. Default: True.

            super_verbose (optional): A bool indicating whether or not to
            print per-iteration updates during each optimization. Default:
            False.

            n_iters_per_print_update (optional): An int specifying how often
            to print updates during the fixed point optimizations. Default:
            100.
        """

        self.dtype = dtype
        self.np_dtype = np.dtype(dtype)
        self.device = next(mrnn.parameters()).device
        self.torch_dtype = getattr(torch, self.dtype)

        # Make random sequences reproducible
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # *********************************************************************
        # Optimization hyperparameters ****************************************
        # *********************************************************************

        self.mrnn = mrnn
        self.lr_init = lr_init
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_cooldown = lr_cooldown
        self.tol_q = tol_q
        self.tol_dq = tol_dq
        self.max_iters = max_iters
        self.do_rerun_q_outliers = do_rerun_q_outliers
        self.outlier_q_scale = outlier_q_scale
        self.do_exclude_distance_outliers = do_exclude_distance_outliers
        self.outlier_distance_scale = outlier_distance_scale
        self.tol_unique = tol_unique
        self.max_n_unique = max_n_unique
        self.verbose = verbose
        self.super_verbose = super_verbose
        self.n_iters_per_print_update = n_iters_per_print_update
        self.batch_first = self.mrnn.batch_first

    # *************************************************************************
    # Primary exposed functions ***********************************************
    # *************************************************************************

    def sample_states(
        self,
        state_traj: torch.Tensor,
        n_inits: int,
        noise_scale: float = 0.0,
        exclude_zero_tensors: bool = False,
    ) -> torch.Tensor:
        """Draws random samples from trajectories of the RNN state. Samples
        can optionally be corrupted by independent and identically distributed
        (IID) Gaussian noise. These samples are intended to be used as initial
        states for fixed point optimizations.

        Args:
            state_traj: 1D or ND tensor containing
            example trajectories of the RNN state.

            n_inits: int specifying the number of sampled states to return.

            noise_scale (optional): non-negative float specifying the standard
            deviation of IID Gaussian noise samples added to the sampled
            states.

            exclude_zero_tensors (bool, optional): whether to exclude zeros
            tensors that may be in state_traj

        Returns:
            initial_states: Sampled RNN states as a [n_inits x n_states] tensor

        Raises:
            ValueError if noise_scale is negative.
        """
        if state_traj.dim() == 1:
            state_traj = state_traj.unsqueeze(0)

        # Get the batch shape of state trajectory, assumes -1 is state dim
        flat_state_traj = torch.flatten(state_traj, end_dim=-2)

        if exclude_zero_tensors:
            non_zero_rows, _ = torch.nonzero(flat_state_traj, as_tuple=True)
            non_zero_rows = torch.unique(non_zero_rows)
            flat_state_traj = flat_state_traj[non_zero_rows]

        rand_indices = torch.randint(high=flat_state_traj.shape[0], size=(n_inits,))
        states = flat_state_traj[rand_indices]

        # Add IID Gaussian noise to the sampled states
        states = self._add_gaussian_noise(states, noise_scale)

        assert not torch.any(torch.isnan(states)), (
            "Detected NaNs in sampled states. Check state_traj and valid_bxt."
        )

        return states

    def find_fixed_points(
        self,
        initial_states: torch.Tensor,
        ext_inputs: torch.Tensor,
        *args,
        stim_inp: torch.Tensor = None,
        W_rec: torch.Tensor = None,
        W_inp: torch.Tensor = None,
        n_rounds_q_opt: int = 1,
    ) -> tuple[FixedPointCollection, FixedPointCollection]:
        """Finds RNN fixed points and the Jacobians at the fixed points.

        Args:
            initial_states: Tensor specifying the initial
            states of the RNN, from which the optimization will search for
            fixed points.

            ext_inputs: external inputs to the RNN

            stim_inp: Additional stimulus input to the network

            W_rec: Fixed weight matrix to replace self.mrnn.W_rec in forward
            pass

            W_rec: Fixed weight matrix to replace self.mrnn.W_inp in forward
            pass

            n_rounds_q_opt: Number of rounds to run extra iterations on q
            outliers

        Returns:
            unique_fps: A FixedPoints object containing the set of unique
            fixed points after optimizing from all initial_states. Two fixed
            points are considered unique if all absolute element-wise
            differences are less than tol_unique AND the corresponding inputs
            are unique following the same criteria. See FixedPoints.py for
            additional detail.

            all_fps: A FixedPoints object containing the likely redundant set
            of fixed points (and associated metadata) resulting from ALL
            initializations in initial_states (i.e., the full set of fixed
            points before filtering out putative duplicates to yield
            unique_fps).
        """

        all_fps = self._fp_optimization(
            initial_states,
            ext_inputs,
            *args,
            stim_inp=stim_inp,
            W_rec=W_rec,
            W_inp=W_inp,
        )

        # Filter out duplicates after from the first optimization round
        unique_fps = all_fps.get_unique()

        self._print_if_verbose("\tIdentified %d unique fixed points." % unique_fps.n)

        if self.do_exclude_distance_outliers:
            unique_fps = self._exclude_distance_outliers(unique_fps, initial_states)

        # Optionally run additional optimization iterations on identified
        # fixed points with q values on the large side of the q-distribution.
        if self.do_rerun_q_outliers:
            unique_fps = self._run_additional_iterations_on_outliers(
                unique_fps,
                stim_inp=stim_inp,
                W_rec=W_rec,
                W_inp=W_inp,
                n_rounds=n_rounds_q_opt,
            )
            # Filter out duplicates after from the second optimization round
            unique_fps = unique_fps.get_unique()

        # Optionally subselect from the unique fixed points (e.g., for
        # computational savings when not all are needed.)
        if unique_fps.n > self.max_n_unique:
            self._print_if_verbose(
                "\tRandomly selecting %d unique "
                "fixed points to keep." % self.max_n_unique
            )
            max_n_unique = int(self.max_n_unique)
            idx_keep = self.rng.choice(unique_fps.n, max_n_unique, replace=False)
            unique_fps = unique_fps[idx_keep]

        self._print_if_verbose("\tFixed point finding complete.\n")

        return unique_fps, all_fps

    # *************************************************************************
    # Helper functions ********************************************************
    # *************************************************************************

    def _add_gaussian_noise(
        self, data: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """Adds IID Gaussian noise to Numpy data.

        Args:
            data: Tensor

            noise_scale: (Optional) non-negative scalar indicating the
            standard deviation of the Gaussian noise samples to be generated.
            Default: 0.0.

        Returns:
            Tensor matching shape of data with noise added

        Raises:
            ValueError if noise_scale is negative.
        """

        # Add IID Gaussian noise
        if noise_scale == 0.0:
            return data  # no noise to add
        if noise_scale > 0.0:
            return data + noise_scale * self.rng.randn(*data.shape)
        elif noise_scale < 0.0:
            raise ValueError(
                "noise_scale must be non-negative, but was %f" % noise_scale
            )

    @staticmethod
    def identify_q_outliers(fps: FixedPointCollection, q_thresh: float) -> torch.Tensor:
        """Identify fixed points with optimized q values that exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.

            q_thresh: A scalar float indicating the threshold on fixed
            points' q values.

        Returns:
            A tensor containing the indices into fps corresponding to
            the fixed points with q values exceeding the threshold.

        Usage:
            idx = identify_q_outliers(fps, q_thresh)
            outlier_fps = fps[idx]
        """
        return torch.where(fps.qstar > q_thresh)[0]

    @staticmethod
    def identify_q_non_outliers(
        fps: FixedPointCollection, q_thresh: float
    ) -> torch.Tensor:
        """Identify fixed points with optimized q values that do not exceed a
        specified threshold.

        Args:
            fps: A FixedPoints object containing optimized fixed points and
            associated metadata.

            q_thresh: A scalar float indicating the threshold on fixed points'
            q values.

        Returns:
            A tensor containing the indices into fps corresponding to the
            fixed points with q values that do not exceed the threshold.

        Usage:
            idx = identify_q_non_outliers(fps, q_thresh)
            non_outlier_fps = fps[idx]
        """
        return torch.where(fps.qstar <= q_thresh)[0]

    @staticmethod
    def get_init_non_distance_outliers(
        initial_states: torch.Tensor, dist_thresh: float
    ) -> torch.Tensor:
        """
        get initial states that are far from centroid based on threshold.
        Args:
            initial_states (Tensor): initial states of fp optimization [n, state_dim]
            dist_thresh (float): Threshold from initial states which is far.
        Returns:
            init_non_outlier_idx (Tensor): indices to initial_states tensor inside threshold
        """

        # Centroid of initial_states, shape (n_states,)
        centroid = torch.mean(initial_states, dim=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = torch.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = torch.mean(init_dists)

        # Normalized distances of initial states to the centroid, shape: (n,)
        scaled_init_dists = torch.true_divide(init_dists, avg_init_dist)

        init_non_outlier_idx = torch.where(scaled_init_dists < dist_thresh)[0]
        return init_non_outlier_idx

    @staticmethod
    def get_fp_non_distance_outliers(
        fps: FixedPointCollection, initial_states: torch.Tensor, dist_thresh: float
    ) -> torch.Tensor:
        """
        get fixed points that are far from initial states based on threshold.
        Args:
            fps (FixedPointCollection): fps discovered [n, state_dim]
            initial_states (Tensor): initial states of optimization [n, state_dim]
            dist_thresh (float): threshold at which fixed points are considered far
        Returns:
            fsp_non_outlier_distance (Tensor): indices to fps object that are not far
        """
        # Centroid of initial_states, shape (n_states,)
        centroid = torch.mean(initial_states, dim=0)

        # Distance of each initial state from the centroid, shape (n,)
        init_dists = torch.linalg.norm(initial_states - centroid, axis=1)
        avg_init_dist = torch.mean(init_dists)

        # Distance of each FP from the initial_states centroid
        fps_dists = torch.linalg.norm(fps.xstar - centroid, axis=1)

        # Normalized
        scaled_fps_dists = torch.true_divide(fps_dists, avg_init_dist)

        fps_non_outlier_idx = torch.where(scaled_fps_dists < dist_thresh)[0]
        return fps_non_outlier_idx

    def _exclude_distance_outliers(
        self, fps: FixedPointCollection, initial_states: torch.Tensor
    ) -> FixedPointCollection:
        """Removes putative distance outliers from a set of fixed points.
        See docstring for identify_distance_non_outliers(...).
        """

        idx_keep = self.get_fp_non_distance_outliers(
            fps, initial_states, self.outlier_distance_scale
        )
        return fps[idx_keep]

    def _run_additional_iterations_on_outliers(
        self,
        fps: FixedPointCollection,
        stim_inp: torch.Tensor = None,
        W_rec: torch.Tensor = None,
        W_inp: torch.Tensor = None,
        n_rounds: int = 1,
    ) -> FixedPointCollection:
        """Detects outlier states with respect to the q function and runs
        additional optimization iterations on those states This should only be
        used after calling either _run_joint_optimization or
        _run_sequential_optimizations.

        Args:
            fps: A FixedPoints object containing (partially) optimized
            fixed points and associated metadata.

            stim_inp: additional stimulus to give network during optimization

            W_rec: replaces self.mrnn.W_rec during forward pass

            W_inp: replaces self.mrnn.W_inp during forward pass

        Returns:
            A FixedPoints object containing the further-optimized fixed points
            and associated metadata.
        """

        """
        Known issue:
            Additional iterations do not always reduce q! This may have to do
            with learning rate schedules restarting from values that are too
            large.
        """

        outlier_min_q = np.median(fps.qstar) * self.outlier_q_scale

        def perform_outlier_optimization(
            fps: FixedPointCollection,
        ) -> FixedPointCollection:
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)

            outlier_fps = fps[idx_outliers]
            n_prev_iters = outlier_fps.n_iters
            inputs = outlier_fps.inputs
            initial_states = outlier_fps.xstar

            self._print_if_verbose(
                "\tPerforming another round of "
                "joint optimization, "
                "over outlier states only."
            )

            updated_outlier_fps = self._fp_optimization(
                initial_states, inputs, stim_inp=stim_inp, W_rec=W_rec, W_inp=W_inp
            )

            updated_outlier_fps.n_iters += n_prev_iters
            fps[idx_outliers] = updated_outlier_fps

            return fps

        def outlier_update(fps: FixedPointCollection) -> torch.Tensor:
            idx_outliers = self.identify_q_outliers(fps, outlier_min_q)
            n_outliers = len(idx_outliers)

            self._print_if_verbose(
                "\n\tDetected %d putative outliers "
                "(q>%.2e)." % (n_outliers, outlier_min_q)
            )

            return idx_outliers

        idx_outliers = outlier_update(fps)

        if len(idx_outliers) == 0:
            return fps

        for _ in range(n_rounds):
            fps = perform_outlier_optimization(fps)
            idx_outliers = outlier_update(fps)
            if len(idx_outliers) == 0:
                return fps

        return fps

    def _print_if_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    @classmethod
    def _print_iter_update(
        cls,
        iter_count: int,
        t_start: float,
        q: torch.Tensor,
        dq: torch.Tensor,
        lr: float,
        is_final: bool = False,
    ):
        t = time.time()
        t_elapsed = t - t_start
        avg_iter_time = t_elapsed / iter_count

        if is_final:
            delimiter = "\n\t\t"
            print("\t\t%d iters%s" % (iter_count, delimiter), end="")
        else:
            delimiter = ", "
            print("\tIter: %d%s" % (iter_count, delimiter), end="")

        if q.size == 1:
            print("q = %.2e%sdq = %.2e%s" % (q, delimiter, dq, delimiter), end="")
        else:
            mean_q = torch.mean(q)
            std_q = torch.std(q)

            mean_dq = torch.mean(dq)
            std_dq = torch.std(dq)

            print(
                "q = %.2e +/- %.2e%s"
                "dq = %.2e +/- %.2e%s"
                % (mean_q, std_q, delimiter, mean_dq, std_dq, delimiter),
                end="",
            )

        print("learning rate = %.2e%s" % (lr, delimiter), end="")

        print("avg iter time = %.2e sec" % avg_iter_time, end="")

    def _fp_optimization(
        self,
        initial_states: torch.Tensor,
        ext_inp: torch.Tensor,
        *args,
        stim_inp: torch.Tensor = None,
        W_rec: torch.Tensor = None,
        W_inp: torch.Tensor = None,
    ) -> FixedPointCollection:
        """Finds multiple fixed points via a joint optimization over multiple
        state vectors.

        Args:
            initial_states: Tensor specifying the initial
            states of the RNN, from which the optimization will search for
            fixed points.

            ext_inp: Tensor specifying a set of constant
            inputs into the RNN.

            stim_inp: Tensor specifying additional stimulus to the network

            W_rec: replaces self.mrnn.W_rec in forward

            W_inp: replaces self.mrnn.W_inp in forward

        Returns:
            fps: A FixedPoints object containing the optimized fixed points
            and associated metadata.
        """

        # Get batch and time dims
        if self.batch_first:
            TIME_DIM = 1
        else:
            TIME_DIM = 0

        initial_states = self._broadcast_nxd(initial_states, tile_n=1)

        # Get batch size of states
        n = initial_states.shape[0]

        # Broadcast external input to [n, 1, d]
        ext_inp = self._broadcast_nxd(ext_inp, tile_n=n)
        ext_inp = ext_inp.unsqueeze(TIME_DIM)

        # Broadcast stimulus input to [n, 1, d]
        if stim_inp is not None:
            stim_inp = self._broadcast_nxd(stim_inp, tile_n=n)
            stim_inp = stim_inp.unsqueeze(TIME_DIM)
        else:
            stim_inp = torch.zeros(size=(n, 1, 1))
            stim_inp = stim_inp.to(self.torch_dtype)
            stim_inp = stim_inp.to(self.device)

        # assert the correct batch shapes
        assert ext_inp.shape[0] == initial_states.shape[0]
        assert stim_inp.shape[0] == initial_states.shape[0]

        self._print_if_verbose(
            "\nSearching for fixed points from %d initial states.\n" % n
        )

        # Ensure that fixed point optimization does not alter RNN parameters.
        print(
            "\tFreezing model parameters so model is not affected by fixed point optimization."
        )

        for p in self.mrnn.parameters():
            p.requires_grad = False

        self._print_if_verbose("\tFinding fixed points via joint optimization.")
        # initialize args to include all regions if empty
        if not args:
            args = list(self.mrnn.region_dict.keys())

        ext_inp.requires_grad = False

        # Gather all of the regions to concatenate during training
        # Get them region by region for proper optimization
        region_tensor_list = []
        region_to_opt_idx = []
        for i, region in enumerate(self.mrnn.region_dict):
            act = self.mrnn.get_region_activity(initial_states, region)
            region_tensor_list.append(act)
            if region in args:
                act.requires_grad = True
                region_to_opt_idx.append(i)

        init_lr = self.lr_init
        optimizer = torch.optim.Adam(
            [region_tensor_list[idx] for idx in region_to_opt_idx], lr=self.lr_init
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.25, patience=10, cooldown=0, threshold=1e-10
        )

        iter_count = 1
        iter_learning_rate = init_lr
        t_start = time.time()
        q_prev_b = torch.full((n,), float("nan"), device=self.device)

        if W_rec is not None:
            W_rec = W_rec.detach().clone()
        if W_inp is not None:
            W_inp = W_inp.detach().clone()

        while True:
            x, h = (
                torch.cat(region_tensor_list, dim=-1),
                torch.cat(region_tensor_list, dim=-1),
            )

            _, F_x_1xbxd = self.mrnn(
                ext_inp,
                x,
                h,
                stim_inp,
                noise=False,
                W_rec=W_rec,
                W_inp=W_inp,
            )
            F_x_1xbxd = F_x_1xbxd.squeeze(TIME_DIM)

            h_prev = []
            h_next = []
            for region in args:
                h_prev.append(self.mrnn.get_region_activity(h, region))
                h_next.append(self.mrnn.get_region_activity(F_x_1xbxd, region))
            h_prev, h_next = torch.cat(h_prev, dim=-1), torch.cat(h_next, dim=-1)

            dx_bxd = h_prev - h_next
            q_b = 0.5 * torch.sum(torch.square(dx_bxd), dim=-1)
            q_scalar = torch.mean(q_b)
            dq_b = torch.abs(q_b - q_prev_b)

            optimizer.zero_grad()
            q_scalar.backward()

            optimizer.step()
            scheduler.step(metrics=q_scalar)

            iter_learning_rate = scheduler.state_dict()["_last_lr"][0]

            ev_q_b = q_b.detach().cpu()
            ev_dq_b = dq_b.detach().cpu()

            if (
                self.super_verbose
                and np.mod(iter_count, self.n_iters_per_print_update) == 0
            ):
                self._print_iter_update(
                    iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate
                )

            if iter_count > 1 and torch.all(
                torch.logical_or(
                    ev_dq_b < self.tol_dq * iter_learning_rate, ev_q_b < self.tol_q
                )
            ):
                """Here dq is scaled by the learning rate. Otherwise very
                small steps due to very small learning rates would spuriously
                indicate convergence. This scaling is roughly equivalent to
                measuring the gradient norm."""
                self._print_if_verbose("\tOptimization complete to desired tolerance.")
                break

            if iter_count + 1 > self.max_iters:
                self._print_if_verbose(
                    "\tMaximum iteration count reached. Terminating."
                )
                break

            q_prev_b = q_b
            iter_count += 1

        if self.verbose:
            self._print_iter_update(
                iter_count, t_start, ev_q_b, ev_dq_b, iter_learning_rate, is_final=True
            )

        # remove extra dims
        # For now make the fixed point include all regions
        full_fp = torch.cat(region_tensor_list, dim=-1)
        xstar = full_fp.detach().cpu()

        F_xstar = F_x_1xbxd.detach().cpu()

        # Indicate same n_iters for each initialization (i.e., joint optimization)
        n_iters = torch.tile(torch.tensor([iter_count]), dims=(F_xstar.shape[0],))
        inputs_bxd = ext_inp.squeeze(TIME_DIM)

        fps = FixedPointCollection(
            xstar=xstar,
            x_init=initial_states,
            inputs=inputs_bxd,
            F_xstar=F_xstar,
            qstar=ev_q_b,
            dq=ev_dq_b,
            n_iters=n_iters,
            tol_unique=self.tol_unique,
            dtype=self.np_dtype,
        )

        return fps

    def _broadcast_nxd(self, data: torch.Tensor, tile_n: int = 1) -> torch.Tensor:
        """
        Takes in a tensor of shape [..., d] and reshapes to nxd
        tiles by tile_n if 1D
        """
        # Broadcast to [n, d]
        if data.dim() == 1:
            # If only 1d, then tile
            data = torch.tile(data, [tile_n, 1])
        else:
            # If > 1d, then flatten up to last dim
            data = torch.flatten(data, end_dim=-2)
        # Ensure proper device and dtype
        data = data.to(self.torch_dtype)
        data = data.to(self.device)
        return data

    def get_jacobian(self, x: torch.Tensor, *args, alpha: float = 1) -> torch.Tensor:
        """
        Wrapper around linearization object jacobian for current fixed points
        """
        linearization = Linearization(self.mrnn)
        return linearization.jacobian(x, *args, alpha=alpha)

    def decompose_jacobian(
        self, x: torch.Tensor, *args, alpha: float = 1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Wrapper around linearization object decomposition for current fixed points
        """
        linearization = Linearization(self.mrnn)
        return linearization.eigendecomposition(x, *args, alpha=alpha)
