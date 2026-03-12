#!/bin/bash
set -e

echo "==========================================================="
echo "Starting the COMPLETE QAG Benchmark Suite (TMLR Submission)"
echo "Outputs will be safely stored in the 'logs/' directory."
echo "==========================================================="

mkdir -p logs

echo "[1/8] Running Section 7.2 (Scaling and Memory Benchmarks)..."
echo "      (Note: Sinkhorn will intentionally trigger an OOM error at N=1,000,000. This is expected.)"
python exp_7_2_scaling.py > logs/results_scaling.txt

echo "[2/8] Running Section 7.3 (Accuracy-Runtime Tradeoff - N=100k)..."
python exp_tradeoff_105.py > logs/results_tradeoff.txt

echo "[3/8] Running Section 7.4 (Naive PyTorch Baselines & Gradient Validation)..."
python exp_new_baselines.py > logs/results_baselines.txt

echo "[4/8] Running POT Forward Exactness Validation..."
python exp_pot_exactness.py > logs/results_pot_exactness.txt

echo "[5/8] Running End-to-End Multi-Seed Tasks (Regression & SW)..."
python exp_multiseed.py > logs/results_multiseed.txt

echo "[6/8] Running Appendix A.1 (Quantile Calibration)..."
python exp_7_5_A_calibration.py > logs/results_appendix_A.txt

echo "[7/8] Running Appendix A.2 (Histogram Mode Preservation)..."
python exp_7_5_B_histogram.py > logs/results_appendix_B.txt

# echo "[8/8] Running Deep Learning MLP Sliced-Wasserstein Task..."
# echo "      (⚠️ Warning: This evaluates Sinkhorn at K=1024. It may take up to 11 hours.)"
# python exp_sliced_wasserstein_mlp.py > logs/results_sw_mlp.txt

echo "==========================================================="
echo "Math computation complete! Generating plots..."
echo "==========================================================="

# Run the plotting script last so it can read the logs we just made
python plot_experiments_7_5.py

echo "All tasks finished successfully. Check the root folder for your new charts!"