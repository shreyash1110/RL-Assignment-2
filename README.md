EE675 Assignment 2 - Policy Gradient for Predator-Prey Game
===========================================================

This folder contains a complete solution for Assignment 2.
The grid size is fixed to N = 4, as required.

Files
-----
environment.py
    Simulator and action-space utilities for the predator-prey game.

policy_network.py
    PyTorch neural network policy pi_theta(a | s).
    Input dimension: 4 normalized coordinates.
    Output dimension: 5 action logits for stay/up/down/left/right.

gradient_estimate.py
    Monte Carlo REINFORCE / score-function gradient estimator.
    Uses torch.distributions.Categorical.log_prob(action) and torch.autograd.grad.

simple_sga.py
    Implements simple stochastic gradient ascent: theta <- theta + eta * g_hat.

train.py
    Runs both simple SGA and Adam, saves logs, and generates learning-curve plots.

assignment2_report.pdf
    Explanation of the design, gradient estimator, optimizers, and generated plots.

requirements.txt
    Minimal package list.

How to run
----------
From this folder, run:

    pip install -r requirements.txt
    python train.py

This creates:

    results/training_log.csv
    plots/simple_sga_learning_curve.png
    plots/adam_learning_curve.png

Useful options
--------------
For a quicker test run:

    python train.py --iterations 20 --episodes 4 --horizon 40

For the run used in the report:

    python train.py --iterations 100 --episodes 4 --horizon 60 --sga-lr 0.02 --adam-lr 0.001

A longer, smoother run can be obtained by increasing --iterations and --episodes.

Notes
-----
The policy network emits logits for all five global actions, but invalid boundary moves are masked before sampling. Therefore the stochastic policy remains a valid distribution over feasible actions at the current predator location.
