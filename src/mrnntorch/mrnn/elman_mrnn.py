"""mRNN core module.

Implements the multi-regional recurrent neural network (mRNN) building blocks
and step-wise dynamics, along with helpers for connectivity, constraints, and
initialization."""

import torch
from typing import Tuple
from mrnntorch.mrnn.mrnn_base import mRNNBase


DEFAULTS_MRNN = {
    "config": None,
    "activation": "relu",
    "noise_level_act": 0.01,
    "noise_level_inp": 0.01,
    "rec_constrained": True,
    "inp_constrained": True,
    "batch_first": True,
    "spectral_radius": None,
    "config_finalize": True,
    "device": "cuda",
}


def linear(x):
    """Return ``x`` unchanged."""
    return x


class ElmanmRNN(mRNNBase):
    """Elman-style multi-regional RNN that evolves only hidden activations."""

    def __init__(
        self,
        config: str = DEFAULTS_MRNN["config"],
        activation: str = DEFAULTS_MRNN["activation"],
        noise_level_act: float = DEFAULTS_MRNN["noise_level_act"],
        noise_level_inp: float = DEFAULTS_MRNN["noise_level_inp"],
        rec_constrained: bool = DEFAULTS_MRNN["rec_constrained"],
        inp_constrained: bool = DEFAULTS_MRNN["inp_constrained"],
        batch_first: bool = DEFAULTS_MRNN["batch_first"],
        spectral_radius: float = DEFAULTS_MRNN["spectral_radius"],
        config_finalize: bool = DEFAULTS_MRNN["config_finalize"],
        device: str = DEFAULTS_MRNN["device"],
    ):
        """Initialize an Elman-style multi-regional RNN.

        Args:
            config (str | None): Optional JSON config path.
            activation (str): Hidden activation function name.
            noise_level_act (float): Hidden-state noise scale.
            noise_level_inp (float): Input noise scale.
            rec_constrained (bool): Whether recurrent weights obey Dale's law.
            inp_constrained (bool): Whether input weights obey Dale's law.
            batch_first (bool): Whether sequences are batch-major.
            spectral_radius (float | None): Optional recurrent spectral-radius target.
            config_finalize (bool): Whether to finalize connectivity after config load.
            device (str): Torch device string.
        """
        super(ElmanmRNN, self).__init__(
            config,
            activation,
            noise_level_act,
            noise_level_inp,
            rec_constrained,
            inp_constrained,
            batch_first,
            spectral_radius,
            config_finalize,
            device,
        )

    def batched_initial_condition(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the batched initial hidden state."""
        hn = self.initial_condition.unsqueeze(0).repeat(batch_size, 1)
        return hn

    def forward(
        self,
        inp: torch.Tensor,
        h0: torch.Tensor,
        stim_input: torch.Tensor | None = None,
        noise: bool = False,
        W_rec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the recurrent dynamics over a sequence.

        Discretized update: ``h_{t+1} =  W_rec h_t + W_inp u_t + b + noise``
        and ``h_{t+1} = activation(x_{t+1})``.

        Args:
            inp (torch.Tensor): Input sequence. Shape ``[B, T, I]`` if batch_first
                else ``[T, B, I]``.
            x0 (torch.Tensor): Initial pre-activation hidden state, shape ``[B, H]``.
            h0 (torch.Tensor): Initial activation, shape ``[B, H]``.
            *args (torch.Tensor): Optional additive inputs with same temporal layout
                as ``inp`` and feature size ``H``.
            noise (bool): If True, add Gaussian noise to hidden state and inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(x_seq, h_seq)`` sequences matching
            the temporal layout of ``inp``.
        """
        assert len(self.region_dict) > 0
        assert len(self.inp_dict) > 0
        assert self.rec_finalized or self.inp_finalized, (
            "Recurrent or input weights are not finalized, \
            call finalize_connectivity() in your custom model definition"
        )

        if inp.dim() != 3:
            raise Exception(
                "input must be 3 dimensional, \
                            [batch, time, units] for batch_first=True, \
                            and [time, batch, units] otherwise]."
            )
        if h0.dim() != 2:
            raise Exception("x0 must be 2 dimensional, [batch, units].")

        if stim_input is not None:
            if stim_input.dim() != 3:
                raise Exception(
                    "stim_input must be 3 dimensional, \
                                [batch, time, units] for batch_first=True, \
                                and [time, batch, units] otherwise]."
                )

        if W_rec is None:
            # Apply Dale's Law if constrained
            if self.rec_constrained:
                W_rec = self.apply_dales_law(
                    self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix
                )
            else:
                W_rec = self.W_rec * self.W_rec_mask
        assert isinstance(W_rec, torch.Tensor)

        # Apply to input weights as well
        if self.inp_constrained:
            W_inp = self.apply_dales_law(
                self.W_inp, self.W_inp_mask, self.W_inp_sign_matrix
            )
        else:
            W_inp = self.W_inp * self.W_inp_mask

        baseline_inp = self.tonic_inp

        hn_next = h0

        if self.batch_first:
            # If batch first then batch is first dim
            batch_shape = inp.shape[0]
            seq_len = inp.shape[1]
            shape = (batch_shape, seq_len, self.total_num_units)
        else:
            # If not batch first then seq_len is first dim
            seq_len = inp.shape[0]
            batch_shape = inp.shape[1]
            shape = (seq_len, batch_shape, self.total_num_units)

        # Create lists for xs and hns
        new_hs = torch.empty(size=shape, device=self.device)

        # Process sequence
        for t in range(seq_len):
            # Gather input at current timestep
            if self.batch_first:
                inp_t = inp[:, t, :]
            else:
                inp_t = inp[t, :, :]

            # Sample from normal distribution and scale by constant term
            if noise:
                # Separate noise levels will be applied to each neuron/input
                hid_noise = self._hid_noise(batch_shape)
                inp_noise = self._inp_noise(batch_shape)
            else:
                hid_noise = inp_noise = 0

            """
            Update hidden state
            Discretized equation of the form: 
                x_(t+1) = x_t + alpha * (-x_t + Wh + W_ix + b)
            """

            xn_next = (
                +(W_rec @ hn_next.T).T
                + (W_inp @ (inp_t + inp_noise).T).T
                + baseline_inp
                + hid_noise
            )

            if stim_input is not None:
                # Add any additional arg inputs (stim inputs typically)
                if self.batch_first:
                    xn_next = xn_next + stim_input[:, t, :]
                else:
                    xn_next = xn_next + stim_input[t, :, :]

            """
            Compute activation
            Activation of the form: 
                h_t = sigma(x_t)
            """

            hn_next = self.activation(xn_next)

            if self.batch_first:
                new_hs[:, t, :] = hn_next
            else:
                new_hs[t, :, :] = hn_next

        return new_hs
