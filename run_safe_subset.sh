#!/bin/bash
set -e
mkdir -p logs

echo "Running Safe Subset (Excluding extreme OOM tests and the 11-hour Sinkhorn MLP)..."
python exp_tradeoff_105.py > logs/results_tradeoff.txt
python exp_new_baselines.py > logs/results_baselines.txt
python exp_pot_heavy.py > logs/results_pot_exactness.txt
python exp_multiseed.py > logs/results_multiseed.txt
python exp_7_5_A_calibration.py > logs/results_appendix_A.txt
python exp_7_5_B_histogram.py > logs/results_appendix_B.txt
python plot_experiments_7_5.py