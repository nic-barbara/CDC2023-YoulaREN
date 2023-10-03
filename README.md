# CDC2023: Nonlinear Youla-REN

This repository contains the code for our paper *Learning Over Contracting and Lipschitz Closed-Loops for Partially-Observed Nonlinear Systems* accepted at CDC 2023. The code is built on the [`RobustNeuralNetworks.jl`](https://github.com/acfr/RobustNeuralNetworks.jl) package which implements the REN models.

This code has been tested with `juilia v1.7.3` and `RobustNeuralNetworks.jl` version `v0.1.0`.

## Installation

Clone this git repository and start the Julia REPL within the project root directory.
```
git clone https://github.com/nic-barbara/CDC2023-YoulaREN.git
cd CDC2023-YoulaREN
```

Start a Julia session, then activate the repository and install dependencies
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
This project depends on a number of larger packages (eg: `Flux.jl`) and an older version of Julia (`v1.7.3`) so installation may take a few minutes.

## Usage

The main scripts used to log experimental data are:

- `src/MagLev/mag_experiment.jl` to train models on magnetic suspension
- `src/QubeServo/qube_experiment.jl` to train models on the rotary-arm pendulum
- Within `src/Robustness/`, run both `mag_adversarial.jl` and `mag_ecrit_save.jl` to generate robustness results on magnetic suspension
- Within `src/Robustness/`, run both `qube_adversarial.jl` and `qube_ecrit_save.jl` to generate robustness results on the rotary-arm pendulum

The main scripts used to visualise results are:

- `src/MagLev/mag_plot_results.jl` to reproduce Fig. 3a in the paper
- `src/QubeServo/qube_plot_results.jl` to reproduce Fig. 3b in the paper
- `src/Robustness/mag_plot_robustness.jl` to reproduce Fig. 4a in the paper
- `src/Robustness/mag_plot_robustness.jl` to reproduce Fig. 4b in the paper

## Contact

For any questions, please contact Nicholas Barbara (nicholas.barbara@sydney.edu.au)
