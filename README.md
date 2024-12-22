# mRNNTorch
***

**Welcome to mRNNTorch!**
This small Python package allows users to effectively build and analyze multi-regional recurrent neural networks (mRNNs) in PyTorch. The `mRNN` module works similar to PyTorch's `nn.RNN` module and can be included in any custom network inheriting from `nn.Module`. Given the complexity of manually building and collecting weight matrices, mRNNTorch provides a solution to automize this process by letting the user specify their network configuration for fast prototyping!

## Usage
***
There are three primary modules a user might need to use, including the `mRNN`, `analysis`, and `utils` modules.
### mRNN
---
To build an mRNN, first build a json configuration file specifying the recurrent regions and connections between regions, in addition to input regions and their connections to recurrent regions. Input regions simply represent a particular input and its connections (weight matrix) to the recurrent regions specified by the user, however mRNNTorch offers multiple inputs to be specified and passed into the nework, each going to a particular set of desired regions. Example configuration files can be found in the `examples` folder. Once a configuration is specified, the `mRNN` module can be used as such
```python
from mRNNTorch.mRNN import mRNN

class MyMRNN(nn.Module):
    def __init__(self, config):
        super(MyMRNN, self).__init__()
        
        self.mrnn = mRNN(config)
    
    forward(self, xn, input):
        xn, hn = self.mrnn(xn, input)
        return hn
```
The full model definition is given as
```python
mRNN(config, activation="relu", noise_level_act=0.01, noise_level_inp=0.01, constrained=True, 
t_const=0.1, batch_first=True, lower_bound_rec=0, upper_bound_rec=10, lower_bound_inp=0, 
upper_bound_inp=10, device="cuda",)
```
The parameter `constrained=True` specifies if Dale's law is to be used by the network, with limits on the weights given by the lower and upper bound parameters.

The forward pass of the model is defined as
```python
forward(xn, *args, noise=True)
```
The parameter `xn` is the preactivation, and an initial activation `hn` will automatically be applied by taking `self.activation(xn)`. An arbitrary amount of arguments can be passed which corresponds to input to the network. The first `N` arguments passed will have weights defined by the first `N` input regions specified in the configuration file. Any other inputs passed into the network after `N` arguments will not have weights, but will still be applied to the network activity. This is useful if additional inputs need to be given for different experiments that don't require input weights, such as optogenetic manipulation.

### Analysis
---
We currently have three analysis tools, including an eigendecomposition of the linearized network, collection of average peri-stimulus time histograms (PSTHs), and flow fields / energy landscapes. The definitions are provided below:
**Note: each of the following definitions requires an mRNN object, not a custom network containing an mRNN object**
```python
linearized_eigendecomposition(mrnn, x, start_region=None, end_region=None, 
start_cell_type=None, end_cell_type=None):
```
This function requires an mRNN object and a set of pre-activation states x (with which a non-linearity will be applied to get hn). This function allows the user to gather a subset of the weights and cell types for analysis as opposed to solely the whole network.

```python
psth(mrnn, act):
```
This function simply takes in an mRNN object and the activity of each region throughout a trajectory and returns the average activity in each individual region.

```python
flow_field(mrnn, timesteps, dimensions, time_skips, num_points, lower_bound_x, upper_bound_x, 
lower_bound_y, upper_bound_y, inp, start_region=None, end_region=None, 
start_region_cell_type=None, end_region_cell_type=None, linearize=False
):
```
Flow field will return a list of velocities for the dimensions specified in addition to the speed of the network at different points on a grid. This can then be used to plot flow fields and energy landscapes using matplotlib. This function requires an mRNN object, the number of timesteps to analyze, the number of dimensions to reduce activity to (using PCA), how many time skips should be included while generating the velocities, number of points on the x and y axis of the low dimensional grid, the bounds of this grid, and the input that will be given to the network to generate a trajectory. Additionally, start and end regions may be specified to gather flow fields for particular regions only. In this case, flow fields for the specified regions will be generated assuming the non-included regions have the activity levels during the trajectory specified by the input.

### Utils
Similar to analysis, there are provided utility functions that take in mRNN objects. The provided utility functions are the following:
* `manipulation_stim`
* `get_region_activity`
* `get_weight_subset`
* `linearize_trajectory`

**This repository is still undergoing major changes, feel free to contribute!**