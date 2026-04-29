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
    Output dimension: 5 action logits, corresponding to stay, up, down, left, and right.

gradient_estimate.py
    Monte Carlo REINFORCE / score-function gradient estimator.
    Uses torch.distributions.Categorical.log_prob(action) to compute log pi_theta(A_t | S_t).
    Uses torch.autograd.grad to obtain the gradient estimate.

simple_sga.py
    Implements simple stochastic gradient ascent:

        theta <- theta + eta * g_hat

train.py
    Runs both simple SGA and Adam, saves logs, and generates learning-curve plots.

assignment2_report.pdf
    Explains the policy design, gradient estimator, optimizer choices, and generated learning curves.

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

For a smoother curve, increase:

    --iterations
    --episodes

Notes
-----
The policy network emits logits for all five global actions, but invalid boundary moves are masked before sampling.

Therefore, the stochastic policy remains a valid distribution over feasible actions at the current predator location:

    sum_{a in A(s)} pi_theta(a | s) = 1

Here, A(s) denotes the set of feasible predator actions at state s.
