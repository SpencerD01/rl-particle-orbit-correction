# rl-particle-orbit-correction

Reinforcement learning for **beam orbit correction in a particle accelerator**.

Trains an agent to set magnetic corrector values that steer a particle beam back onto
its target trajectory, using the Bmad accelerator physics simulator as the environment.

**Result: maximum orbit deviation reduced from 17mm to 5mm, with improved
run-to-run consistency.**

## Why this is hard

Orbit correction is a continuous, high-dimensional control problem where the action
space (corrector magnet strengths) interacts non-locally with the observation space
(beam position monitors along the ring) — nudging one magnet perturbs the orbit
everywhere downstream. Classical response-matrix inversion handles the linear regime
well; the question here was whether a learned policy could do better without an
explicit model.

## Approach

- **Environment** — a Gymnasium-compatible wrapper around Bmad (via PyTao). State is
  the measured orbit; actions set corrector magnet values; reward is a configurable
  function of deviation from the target orbit
- **Algorithms** — compared SAC, PPO, and TD3 via Stable Baselines
- **Tuning** — Bayesian optimization over the hyperparameter space, which mattered
  more than the choice of algorithm did

## Layout

| File | Purpose |
|---|---|
| `environ.py` | Accelerator environment — generates state via PyTao, applies actions, computes reward |
| `orbit_correctionv1.ipynb` | Main training notebook |
| `hyperparameter_tuning.ipynb` | Bayesian optimization over hyperparameters |
| `Lattice Files/` | Accelerator configuration; runs the Bmad simulation and reports orbit values |
| `pytao_test.ipynb` | Scratch notebook for the PyTao simulation and data-retrieval API |
| `stable_baselines_test.ipynb` | Sanity check of the RL package on a toy problem |

## Running it

The notebooks are written for Google Colab and benefit from GPU/TPU access. To run
locally, skip the Colab install block and substitute `pip install` for the
`conda install` cells.

Requires: `stable-baselines3`, `gymnasium`, `pytao` (Bmad).

## Context

Built for CS 4701 (Practicum in Artificial Intelligence) at Cornell, Spring 2024,
against Cornell's linear accelerator lattice.
