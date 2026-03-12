#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "==========================================================="
echo "Starting the QAG vs. Sinkhorn Benchmark Suite"
echo "Outputs will be safely stored in the 'logs/' directory."
echo "==========================================================="

# Create a clean folder for outputs so we don't pollute the repo
mkdir -p logs

echo "[1/6] Running Section 7.2 (Scaling and Memory Benchmarks)..."
echo "      (Note: Sinkhorn will intentionally trigger an OOM error at N=1,000,000. This is expected.)"
python exp_7_2_scaling.py > logs/results_scaling.txt
echo "      -> Saved to logs/results_scaling.txt"

echo "[2/6] Running Section 7.3 (Accuracy-Runtime Tradeoff - N=100k)..."
echo "      (⏳ Warning: Computing the Sinkhorn blur sweep at N=100,000 takes about 2 to 3 minutes.)"
python exp_tradeoff_105.py > logs/results_tradeoff.txt
echo "      -> Saved to logs/results_tradeoff.txt"

echo "[3/6] Running Section 7.4 (Naive PyTorch Baselines & Gradient Validation)..."
python exp_new_baselines.py > logs/results_baselines.txt
echo "      -> Saved to logs/results_baselines.txt"

echo "[4/6] Running POT Forward Exactness Validation..."
python exp_pot_exactness.py > logs/results_pot_exactness.txt
echo "      -> Saved to logs/results_pot_exactness.txt"

echo "[5/6] Running End-to-End Multi-Seed Tasks (Regression & SW)..."
python exp_multiseed.py > logs/results_multiseed.txt
echo "      -> Saved to logs/results_multiseed.txt"

echo "[6/6] Running Deep Learning MLP Sliced-Wasserstein Task..."
echo "      (⚠️ Warning: This evaluates Sinkhorn at K=1024. It may take several hours.)"
python exp_sliced_wasserstein_mlp.py > logs/results_sw_mlp.txt
echo "      -> Saved to logs/results_sw_mlp.txt"

echo "==========================================================="
echo "All main benchmarks completed successfully!"
echo "==========================================================="