#!/bin/sh

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

# Tell Luigi to queue various simulations based on a model's predictions
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Predictions \
    --ads-list '["CO"]' \
    --prediction-min -2.6 \
    --prediction-max 1.4 \
    --prediction-target -0.60 \
    --model-location '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --priority 'gaussian' \
    --block '("CO",)' \
    --xc 'rpbe' \
    --max-submit 200 \
    --scheduler-host $luigi_port \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300 
