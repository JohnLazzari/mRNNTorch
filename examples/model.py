import torch.nn as nn
from mrnntorch import mRNN, ElmanmRNN


class LeakmRNN(nn.Module):
    def __init__(self, inp_dim, exc_units, inhib_units, out_dim, device="cuda"):
        super().__init__()

        self.rnn = mRNN(inp_constrained=False, dt=1, tau=2, spectral_radius=1, device=device, activation="tanh")
        self.rnn.add_recurrent_region("exc", exc_units, "pos", learnable_bias=True)
        self.rnn.add_recurrent_region("inhib", inhib_units, "neg", learnable_bias=True)
        self.rnn.add_input_region("inp", inp_dim)

        self.rnn.add_recurrent_connection("exc", "exc", sparsity=0.9)
        self.rnn.add_recurrent_connection("exc", "inhib")
        self.rnn.add_recurrent_connection("inhib", "exc")
        self.rnn.add_recurrent_connection("inhib", "inhib")

        self.rnn.add_input_connection("inp", "exc")
        self.rnn.add_input_connection("inp", "inhib")

        self.rnn.finalize_connectivity()

        self.out_layer = nn.Linear(self.rnn.total_num_units, out_dim)

    def forward(self, input, x, h):
        x, h = self.rnn(input, x, h, noise=False)
        out = self.out_layer(h)
        return out, x, h


class EmRNN(nn.Module):
    def __init__(self, inp_dim, exc_units, inhib_units, out_dim, device="cuda"):
        super().__init__()

        self.rnn = ElmanmRNN(inp_constrained=False, spectral_radius=1, device=device, activation="tanh")
        self.rnn.add_recurrent_region("exc", exc_units, "pos", learnable_bias=True)
        self.rnn.add_recurrent_region("inhib", inhib_units, "neg", learnable_bias=True)
        self.rnn.add_input_region("inp", inp_dim)

        self.rnn.add_recurrent_connection("exc", "exc", sparsity=0.9)
        self.rnn.add_recurrent_connection("exc", "inhib")
        self.rnn.add_recurrent_connection("inhib", "exc")
        self.rnn.add_recurrent_connection("inhib", "inhib")

        self.rnn.add_input_connection("inp", "exc")
        self.rnn.add_input_connection("inp", "inhib")

        self.rnn.finalize_connectivity()

        self.out_layer = nn.Linear(self.rnn.total_num_units, out_dim)

    def forward(self, input, hx):
        h = self.rnn(input, hx, noise=False)
        out = self.out_layer(h)
        return out, h
