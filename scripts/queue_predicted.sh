#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=regular
#SBATCH --job-name=queue_predicted
#SBATCH --output=queue_predicted-%j.out
#SBATCH --error=queue_predicted-%j.error
#SBATCH --constraint=haswell

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"
luigi_port="$(python .read_rc.py luigi_port)"

# Load the appropriate environment, etc.
module load python
cd GASpy_feedback/gaspy_feedback
source activate $conda_path

# CO2RR:  Tell Luigi to queue various simulations based on a model's predictions
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Predictions \
    --ads-list '["CO"]' \
    --prediction-min -2.6 \
    --prediction-max 1.4 \
    --prediction-target -0.60 \
    --predictions-location '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --priority 'gaussian' \
    --block '("CO",)' \
    --xc 'rpbe' \
    --max-submit 350 \
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300 

# HER:  Tell Luigi to queue various simulations based on a model's predictions
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Predictions \
    --ads-list '["H"]' \
    --prediction-min -2.28 \
    --prediction-max 1.72 \
    --prediction-target -0.28 \
    --predictions-location '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --priority 'gaussian' \
    --block '("H",)' \
    --xc 'rpbe' \
    --max-submit 350 \
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300 
