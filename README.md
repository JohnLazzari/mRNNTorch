# mRNNTorch
mRNNTorch is a small Python package for building and analyzing multi-regional recurrent neural networks (mRNNs) in PyTorch. The `mRNN` module mirrors `torch.nn.RNN` usage while managing region-level connectivity, masks, and constraints so you can prototype quickly.

## Highlights
- Define recurrent and input regions in JSON, Python, or a mix of both
- Automatic assembly of block weight matrices and masks
- Optional sign constraints (Dale's Law) and spectral radius scaling
- Analysis helpers for fixed points, flow fields, and linearization

## Package layout
| Component | Description |
| --- | --- |
| `mrnntorch.mRNN` | Core mRNN module for building custom networks |
| `mrnntorch.region` | Region classes and connectivity primitives |
| `mrnntorch.analysis` | Analysis utilities operating on an mRNN instance |

## Installation
Install from source:

```bash
python -m pip install -e .
```

## Quick start
There are two ways to define an mRNN:
1. Pass a JSON config file for regions and connections.
2. Build regions and connections manually inside your model.

You can also mix both approaches, then finalize connectivity:

```python
import torch.nn as nn
from mrnntorch.mRNN import mRNN

class MyMRNN(nn.Module):
    def __init__(self, config_path: str):
        super().__init__()
        self.mrnn = mRNN(config_path, config_finalize=False)

        # Add any extra recurrent connections not in the config
        regions = ["r1", "r2"]
        for src in regions:
            for dst in regions:
                self.mrnn.add_recurrent_connection(src, dst)

        # Required if you manually add regions/connections
        self.mrnn.finalize_connectivity()

    def forward(self, inp, x, h):
        return self.mrnn(inp, x, h)
```

## Configuration files
The config format specifies recurrent/input regions and their connections. See `src/mrnntorch/examples/configurations/CBGTCL.json` for a concrete example.

## Requirements
Runtime dependencies:
- torch
- numpy
- matplotlib
- scikit-learn (analysis helpers)

## Status
This repository is still under active development; contributions are welcome.
