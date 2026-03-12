# Quantile Affine Geometry (QAG) for 1D Optimal Transport

This repository contains the official PyTorch implementation and experimental reproducible scripts for the paper evaluating **Quantile Affine Geometry (QAG)** as an exact, GPU-resident 1D Optimal Transport primitive.

## Repository Structure

### Core Library
* `losses.py`: Contains the core `QAGLoss` implementation and the `SinkhornLoss` baseline wrapper (via GeomLoss).

### Main Paper Experiments
* `exp_7_2_scaling.py`: Systems benchmark evaluating forward/backward runtime and memory consumption across varying sample sizes.
* `exp_tradeoff_105.py`: Evaluates the accuracy-runtime tradeoff of entropic OT (Sinkhorn) against the exact QAG reference.
* `exp_new_baselines.py`: Contains comparisons against a naive PyTorch exact sort baseline and the finite-difference directional gradient check.
* `exp_pot_exactness.py`: Forward correctness validation against the POT (Python Optimal Transport) exact 1D reference across continuous, bimodal, and heavy-tailed distributions.
* `exp_multiseed.py`: End-to-end task (Distributional Regression) evaluated over 5 random seeds to measure training convergence and runtime.
* `exp_sliced_wasserstein_mlp.py`: End-to-end Deep Learning task embedding inputs into a class-specific Gaussian Mixture target using a many-1D Sliced-Wasserstein objective. 

### Appendix Experiments
* `exp_7_5_A_calibration.py`: OT-based quantile calibration mapping.
* `exp_7_5_B_histogram.py`: Mode preservation for multi-modal feature distributions.
* `plot_experiments_7_5.py`: Generates the comparative visual plots for the end-to-end tasks.

## Requirements
* `torch` (>= 2.0.0, compiled with CUDA)
* `geomloss` (for Sinkhorn baselines)
* `POT` (Python Optimal Transport, for CPU exactness references)
* `numpy`
* `matplotlib`

## Reproducing the Results
You can run the full suite of experiments sequentially using the provided bash script:
```bash
bash run_experiments.sh
```

Note: The Deep Learning Sliced-Wasserstein experiment (exp_sliced_wasserstein_mlp.py) evaluates Sinkhorn at up to K=1024 slices. Due to the entropic baseline's computational bottleneck at high slice counts, this script may take several hours to complete over 20 epochs. QAG executes this same loop natively in seconds.

***

