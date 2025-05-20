# mRNNTorch
**Welcome to mRNNTorch!**
This small Python package allows users to effectively build and analyze multi-regional recurrent neural networks (mRNNs) in PyTorch. The `mRNN` module works similar to PyTorch's `nn.RNN` module and can be included in any custom network inheriting from `nn.Module`. Given the complexity of manually building and collecting weight matrices, mRNNTorch provides a solution to automize this process by letting the user specify their network configuration for fast prototyping!

## mRNNTorch Structure

|     Component      |                          Description                                       |
|     ---------      | --------------------------------------------------------------------------  |
| mRNNTorch.mRNN     | Defines the mRNN module to use in custom network inheriting from nn.Module  |
| mRNNTorch.analysis | Analysis tools that work directly on an mRNN object to explore its dynamics |
| mRNNTorch.Region   | Region class that defines the properties of a region used in the mRNN object|

## Usage
### mRNN
There are two primary ways to build an mRNN using mRNNTorch

1. Specify a json configuration file with recurrent and input regions and connections
2. Manually design the mRNN within your own custom model

Additionally, users can mix and match approaches. Json files are convenient for saving previous model configuations and easily reusing, however manually entering many connections can be cumbersome. Defining connections in your own custom model may be less flexible across configurations, but can allow for ease of model definition.

Below we provide an example use case:

```python
from mRNNTorch.mRNN import mRNN

# Mix of config and manual entry
class MyMRNN(nn.Module):
    def __init__(self, config):
        super(MyMRNN, self).__init__()
        
        """
            Here, our config defines our regions and
            the input connections
            If we don't define recurrent connections in 
            the config, we can still enter them here.
        """
        self.mrnn = mRNN(config)

        connections = ["r1", "r2"]

        for src_region in connection_props:
            for dst_region in connection_props:
                self.mrnn.add_recurrent_connection(
                    src_region,
                    dst_region
                )

        """
            Whenever defining connectivity or regions outside of the config,
            we must use finalize_connectivity() in order to pad unconnected regions
            with zeros. Otherwise, an error will occur.

            If all regions and connections are defined in the config, the model will
            automatically finalize connectivity unless config_finalize is set to False.
        """

        self.mrnn.finalize_connectivity()
    
    forward(self, inp, x, h):
        xn, hn = self.mrnn(inp, x, h)
        return xn, hn
```

## Requirements

The following packages are necessary to use the mRNNTorch package:

* torch
* numpy
* matplotlib
* sklearn (for analysis module)

**This repository is still undergoing major changes, feel free to contribute!**