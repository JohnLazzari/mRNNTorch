# mRNNTorch
**Welcome to mRNNTorch!**
This small Python package allows users to effectively build and analyze multi-regional recurrent neural networks (mRNNs) in PyTorch. The `mRNN` module works similar to PyTorch's `nn.RNN` module and can be included in any custom network inheriting from `nn.Module`. Given the complexity of manually building and collecting weight matrices, mRNNTorch provides a solution to automize this process by letting the user specify their network configuration for fast prototyping!

## Usage
### mRNN
There are two primary ways to build an mRNN using mRNNTorch

1. Specify a json configuration file with recurrent and input regions and connections
2. Manually design the mRNN within your own custom model

Additionally, users can mix and match approaches. Json files are convenient for saving previous model configuations and easily reusing, however manually entering many connections can be cumbersome. Defining connections in your own custom model may be less flexible across configurations, but can allow for ease of model definition.

**Note:** When passing a configuration into the model, both recurrent and input regions must be specified at the minimum. 

Below we provide example usage cases:
```python
from mRNNTorch.mRNN import mRNN

# Only using configuration
class MyMRNN(nn.Module):
    def __init__(self, config):
        super(MyMRNN, self).__init__()
        
        """
            Model is fully defined by configuration
            including regions and connections
        """
        self.mrnn = mRNN(config)
    
    forward(self, xn, inp):
        xn, hn = self.mrnn(xn, inp)
        return hn
```

```python
from mRNNTorch.mRNN import mRNN

# Only using manual entry
class MyMRNN(nn.Module):
    def __init__(self, input_units, r1_units, r2_units):
        super(MyMRNN, self).__init__()
        
        # pass in any other parameters as needed
        self.mrnn = mRNN()

        # Add recurrent and input regions
        self.mrnn.add_recurrent_region("r1", r1_units)
        self.mrnn.add_recurrent_region("r2", r2_units)
        self.mrnn.add_input_region("input", input_units)

        # Add connections between regions
        self.mrnn.add_input_connection("input", "r1")
        self.mrnn.add_input_connection("input", "r2")

        """
            As opposed to manually entering four connections
            we can loop through this process dynamically
            This is beneficial when the number of connections 
            grows very large
        """

        # Assuming r1 is excitatory and r2 inhibitory
        # For an unconstrained network sign will be ignored
        connection_props = {"r1": {"sign": "exc"}, 
                            "r2": {"sign": "inhib"}}
        for src_region in connection_props:
            for dst_region in connection_props:
                self.mrnn.add_recurrent_connection(
                    src_region,
                    dst_region,
                    sign=connection_props[src_region]["sign"]
                )
        # This is necessary after manually defining connections
        # Using finalize_connections() will pad undefined 
        # connections between regions with zeros
        # Otherwise, you will run into an error
        self.mrnn.finalize_connectivity()
    
    forward(self, xn, inp):
        xn, hn = self.mrnn(xn, inp)
        return hn
```

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

        connection_props = {"r1": {"sign": "exc"}, 
                            "r2": {"sign": "inhib"}}

        for src_region in connection_props:
            for dst_region in connection_props:
                self.mrnn.add_recurrent_connection(
                    src_region,
                    dst_region,
                    sign=connection_props[src_region]["sign"]
                )
        self.mrnn.finalize_connectivity()
    
    forward(self, xn, inp):
        xn, hn = self.mrnn(xn, inp)
        return hn
```

The forward pass of the model is defined as

```python
forward(xn, inp, *args, noise=True)
```

The parameter `xn` is the preactivation, and an initial activation `hn` will automatically be a copy of `xn`. The second argument corresponds to the network input of shape `[batch, length, neurons]` for `batch_first = True` and the alternative otherwise. An arbitrary amount of arguments can be passed which corresponds to additional input to the network not containing weights. 

**This repository is still undergoing major changes, feel free to contribute!**