```{toctree}
:hidden:
:caption: Get Started

readme
```

```{toctree}
:hidden:
:caption: API Reference

api/index
```

:::{container} hero-panel
::::{grid} 1 1 2 2
:class-container: hero-grid
:gutter: 2

:::{grid-item}
:class: hero-copy

<p class="hero-kicker">Multi-regional RNN toolkit</p>

# mRNNTorch

<p class="hero-subtitle">Build, run, and analyze multi-regional recurrent neural networks in PyTorch. Define region connectivity, enforce Dale's law, and probe dynamics with fixed-point and flow-field tools.</p>

:::{container} hero-cta
:::{button-ref} readme
:color: primary
:class: hero-button
Get started
:::
:::{button-ref} api/index
:color: secondary
:class: hero-button ghost
API reference
:::
:::

:::{container} hero-highlights
:::{container} hero-highlight
**Config-first**  
JSON, Python, or mixed region graphs
:::
:::{container} hero-highlight
**Constrained weights**  
Dale's law masks and spectral scaling
:::
:::{container} hero-highlight
**Dynamics ready**  
Fixed points, flow fields, linearization
:::
:::
:::

:::{grid-item}
:class: hero-art

<img alt="mRNNTorch hero" src="_static/hero.svg" class="hero"/>

:::{container} hero-code
```python
from mrnntorch.mRNN import mRNN

mrnn = mRNN("configs/regions.json", config_finalize=False)
mrnn.add_recurrent_connection("PFC", "M1")
mrnn.finalize_connectivity()
```
:::
:::
::::
:::

## Build blocks

::::{grid} 1 1 3 3
:class-container: feature-grid
:gutter: 2

:::{grid-item-card} Regions
:class: feature-card
Define recurrent and input regions with sizes, signs, tonic inputs, and learnable biases.
:::

:::{grid-item-card} Connectivity
:class: feature-card
Compose block weight matrices, enforce Dale's law masks, and scale spectral radius.
:::

:::{grid-item-card} Analysis
:class: feature-card
Find fixed points, visualize flow fields, and linearize dynamics for stability insights.
:::
::::

## Workflow

::::{grid} 1 1 3 3
:class-container: workflow-grid
:gutter: 2

:::{grid-item-card} 1. Configure
:class: workflow-step
Start from JSON configs or manual region graphs.
:::

:::{grid-item-card} 2. Assemble
:class: workflow-step
Finalize connectivity and constraints when the topology is ready.
:::

:::{grid-item-card} 3. Probe
:class: workflow-step
Run analyses to inspect attractors, trajectories, and linear modes.
:::
::::

## Keep exploring

::::{grid} 1 1 2 2
:class-container: explore-grid
:gutter: 2

:::{grid-item-card} Quick start
:link: readme
:class: explore-card
:text-align: left
Install dependencies, build your first network, and run a forward pass.
:::

:::{grid-item-card} API reference
:link: api/index
:class: explore-card
:text-align: left
Detailed docs for `mRNN`, `Region`, and analysis modules.
:::
::::
