.. mRNNTorch documentation master file, created by
   sphinx-quickstart on Thu Feb 19 02:23:50 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mRNNTorch
=========

Effectively design, train, and analyze multi-regional recurrent neural networks in pytorch.

Highlights
----------

- Flexible: design models with json files or manually in a custom model
- Specialized Analysis: custom analysis module allows for targeting particular sets of regions 
- Built with Pytorch: easily extends to common pytorch functionality

Quickstart
----------

Download the repository and pip install in root:

.. code-block:: bash

   pip install -e .

import and use in custom rnn model:

.. code-block:: python

   import mrnntorch
   from mrnntorch.mrnn import mRNN

   class RNN(nn.Module):
      def __init__(self, config):
        super().__init__()
        self.mrnn = mRNN(config)

      def forward(self, input, x, h):
        xn, hn = self.mrnn(input, x, h)
        return xn, hn

Documentation
-------------


.. note::

   This project is under active development. If something is unclear, open an issue.

.. toctree::
  :maxdepth: 1
  
  mrnntorch
