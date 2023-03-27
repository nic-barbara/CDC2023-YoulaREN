# CDC2023: Nonlinear Youla-REN

This repository contains the code for our paper *Learning Over All Contracting and Lipschitz Closed-Loops for Partially-Observed Nonlinear Systems* submitted to CDC 2023. The code is built on the `RobustNeuralNetworks.jl` package **[provide link when public]** which implements the REN models.

This code has been tested with `juilia v1.7.3` and `RobustNeuralNetworks.jl` version `v0.1.0`.

## Installation

1. Clone this git repository and start the Julia REPL within the project root directory. 

2. Activate the repository. Enter the package manager by typing `]` in the REPL, then type `activate .` This may produce an error about `RobustNeuralNetworks.jl` not being registered. Proceed to the next step.

3. Our models depend on the `RobustNeuralNetworks.jl` package, which is currently unregistered and must be added separately. Within the package manager, type `add git@github.com:acfr/RobustNeuralNetworks.jl.git` to add the package.

4. Instantiate the rest of the project. Type `instantiate` within the package manager

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