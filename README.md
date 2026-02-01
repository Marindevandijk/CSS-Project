# CSS-Project
Github for project Complex System Simulation.

This project studies a network model of belief dynamics to understand how interactions lead to consensus or fragmentation. We simulate the model and analyze community sizes as system parameters are varied.

## Library structure

| Module / Folder              | Purpose                                                                 |
| ---------------------------- | ----------------------------------------------------------------------- |
| `base_model.py`          | Core belief-dynamics models (homogeneous and heterogeneous variants)       |
| `run_experiments.py`      | Runs simulations to generate p(s) distributions and data collapse          |
| `Critical_fraction.py`       | Computes and plots max community size and consensus size vs mediator fraction |
| `helpers.py`       | Serves as the mathematical engine for analysis and handling statistical tasks  |
| `plot_experiments.py`     | Visualization of p(s) distributions and data collapse results           |
| `Extras/`     | Contains Roza's initial alternate test version of the base model, unused because we continued with better base_model.py, but included for the sake of contributions. Also it contains the extension of base model to include social media influence but unused because of lack of time.   |
| `Complex Systems Presentation.pptx`            | Power point presentation     |
| `data/`            | Stored data file      |
| `figures/`               | Containts the generated figures for analysis             |


## Requirements 
This package is written in Python 3 and requires the following packages:

- `numpy`
- `networkx`
- `scipy`
- `zlib`
- `powerlaw`
  
### Optional (for plotting and saving results)
- `matplotlib`
- `csv` (part of the Python standard library)

## References 
* **Coevolving networks:** P. Holme & M. E. Newman, *Nonequilibrium phase transition in the coevolution of networks and opinions*, **Phys. Rev. E 74**, 056108 (2006).
