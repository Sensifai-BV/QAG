#!/bin/bash
set -e
mkdir -p logs

echo "Running Full Main Paper Suite (Warning: Will trigger OOM errors and take >11 hours)..."
python exp_7_2_scaling.py > logs/results_scaling.txt
python exp_tradeoff_105.py > logs/results_tradeoff.txt
python exp_new_baselines.py > logs/results_baselines.txt
python exp_pot_heavy.py > logs/results_pot_exactness.txt
python exp_multiseed.py > logs/results_multiseed.txt
python exp_7_5_A_calibration.py > logs/results_appendix_A.txt
python exp_7_5_B_histogram.py > logs/results_appendix_B.txt
python exp_sliced_wasserstein_mlp.py > logs/results_sw_mlp.txt
python plot_experiments_7_5.py