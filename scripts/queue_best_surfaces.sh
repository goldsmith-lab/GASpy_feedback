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

# Use Luigi to queue various surfaces for simulation
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["CO"]' \
    --performance-threshold 0.1 \
    --max-surfaces 1 \
    --scheduler-host $luigi_port \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
