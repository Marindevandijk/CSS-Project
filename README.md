# CSS-Project
Github for project Complex System Simulation.

This project studies a network model of belief dynamics to understand how interactions lead to consensus or fragmentation. We simulate the model and analyze community sizes as system parameters are varied.

## Library structure

| Module / Folder              | Purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| `base_model.py`          | Core belief-dynamics models (homogeneous and heterogeneous variants)       |
| `run_experiments.py`      | Runs simulations to generate p(s) distributions and data collapse          |
| `Critical_fraction.py`       | Computes and plots max community size and consensus size vs mediator fraction |
| `plot_experiments.py`     | Visualization of p(s) distributions and data collapse results           |
| `data/`            | Stored data file      |
| `figures/`               | Containts the generated figures for analysis             |


## Requirements 
This package is written in Python 3 and requires the following packages:

- `numpy`
- `networkx`

### Optional (for plotting and saving results)
- `matplotlib`
- `csv` (part of the Python standard library)

## References 
* **Coevolving networks:** P. Holme & M. E. Newman, *Nonequilibrium phase transition in the coevolution of networks and opinions*, **Phys. Rev. E 74**, 056108 (2006).
