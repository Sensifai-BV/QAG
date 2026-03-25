# Quantile Affine Geometry (QAG) for 1D Optimal Transport

This repository contains the official PyTorch implementation and experimental reproducible scripts for the paper evaluating **Quantile Affine Geometry (QAG)** as an exact, GPU-resident 1D Optimal Transport primitive.

## Repository Structure

### Core Library
* `losses.py`: Contains the core `QAGLoss` implementation and the `SinkhornLoss` baseline wrapper (via GeomLoss).

### Main Paper Experiments
* `exp_7_2_scaling.py`: Systems benchmark evaluating forward/backward runtime and memory consumption across varying sample sizes.
* `exp_tradeoff_105.py`: Evaluates the accuracy-runtime tradeoff of entropic OT (Sinkhorn) against the exact QAG reference.
* `exp_new_baselines.py`: Contains comparisons against a naive PyTorch exact sort baseline and the finite-difference directional gradient check.
* `exp_pot_heavy.py`: Forward correctness validation against the POT (Python Optimal Transport) exact 1D reference across continuous, bimodal, and heavy-tailed distributions.
* `exp_multiseed.py`: End-to-end task (Distributional Regression) evaluated over 5 random seeds to measure training convergence and runtime.
* `exp_sliced_wasserstein_mlp.py`: End-to-end Deep Learning task embedding inputs into a class-specific Gaussian Mixture target using a many-1D Sliced-Wasserstein objective. 

**The Context:**
Section 7.2 of the paper demonstrates the extreme memory bottleneck of standard entropic Optimal Transport (Sinkhorn) on a GPU. At $N=1,000,000$ with a batch size of 64, the tensorized backend requires allocating a dense distance matrix that mathematically demands hundreds of Terabytes of memory.

**The Hardware Reality:**
When this allocation is requested on an NVIDIA L4 GPU (24GB VRAM limit), PyTorch should theoretically issue an immediate `CUDA OutOfMemoryError`. However, depending on the host OS configuration, Linux memory managers will often attempt to page this massive allocation request into system RAM and disk swap space. This results in severe "swap thrashing," causing the system to hang entirely for several hours before finally successfully terminating with the expected OOM error. 

**How to Test Safely:**
To verify the QAG scaling metrics and the Sinkhorn VRAM bottleneck without locking your system, you can safely uncomment Step 1 in the bash script and modify `exp_7_2_scaling.py` to evaluate up to $N=100,000$ (which safely triggers a rapid VRAM OOM on most modern GPUs) instead of $N=1,000,000$. QAG will execute smoothly at all scales.

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
The repository provides two execution paths to accommodate hardware constraints:

**1. The Safe Subset (Recommended for Evaluators):**
Runs the core benchmarks, tradeoff metrics, baselines, fast regression tasks, and plotting tools in under 5 minutes.
`bash run_safe_subset.sh`

**2. The Full Suite (Complete Reproduction):**
Executes every script in the paper sequentially. 
`bash run_full_suite.sh`

### ⚠️ Hardware Assumptions & The Memory Wall
The Full Suite executes Section 7.2 (`exp_7_2_scaling.py`) which is mathematically designed to trigger a `CUDA OutOfMemoryError` on GPUs with <= 24GB of VRAM (e.g., NVIDIA L4 or RTX 3090/4090). 
Depending on your host operating system and swap space configuration, attempting to allocate Sinkhorn's $1,000,000 \times 1,000,000$ dense distance matrix may cause severe system hang or disk swap thrashing before correctly terminating. The Full Suite also runs the Sliced-Wasserstein deep learning baseline, which takes ~11 hours to compute Sinkhorn sequentially at $K=1024$ projections. This experiment is removed in un_safe_subset script.


Note: The Deep Learning Sliced-Wasserstein experiment (exp_sliced_wasserstein_mlp.py) evaluates Sinkhorn at up to K=1024 slices. Due to the entropic baseline's computational bottleneck at high slice counts, this script may take several hours to complete over 20 epochs. QAG executes this same loop natively in seconds.

***

