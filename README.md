# EE675 Assignment 2 - Policy Gradient for Predator-Prey Game

This folder contains a complete solution for Assignment 2.  
The grid size is fixed to **\( N = 4 \)**, as required.

---

## Files

### `environment.py`
Simulator and action-space utilities for the predator-prey game.

### `policy_network.py`
PyTorch neural network policy \( \pi_\theta(a \mid s) \).  
- Input dimension: 4 normalized coordinates  
- Output dimension: 5 action logits (stay/up/down/left/right)

### `gradient_estimate.py`
Monte Carlo **REINFORCE / score-function gradient estimator**.  
Uses:
- `torch.distributions.Categorical.log_prob(action)`
- `torch.autograd.grad`

### `simple_sga.py`
Implements simple stochastic gradient ascent:

\[
\theta \leftarrow \theta + \eta \cdot \hat{g}
\]

### `train.py`
Runs both **Simple SGA** and **Adam**, saves logs, and generates learning-curve plots.

### `assignment2_report.pdf`
Explanation of:
- Design
- Gradient estimator
- Optimizers
- Generated plots

### `requirements.txt`
Minimal package list.

---

## How to Run

From this folder:

```bash
pip install -r requirements.txt
python train.py
```

---

## Outputs

Running the script generates:

- `results/training_log.csv`
- `plots/simple_sga_learning_curve.png`
- `plots/adam_learning_curve.png`

---

## Useful Options

### Quick Test Run
```bash
python train.py --iterations 20 --episodes 4 --horizon 40
```

### Run Used in Report
```bash
python train.py --iterations 100 --episodes 4 --horizon 60 --sga-lr 0.02 --adam-lr 0.001
```

### Longer / Smoother Training
Increase:
- `--iterations`
- `--episodes`

---

## Notes

The policy network emits logits for all five global actions, but **invalid boundary moves are masked before sampling**.  

Therefore, the stochastic policy remains a valid distribution over feasible actions at the current predator location:

\[
\sum_{a \in \mathcal{A}(s)} \pi_\theta(a \mid s) = 1
\]
