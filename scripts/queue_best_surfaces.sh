#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=queue_best_surfaces
#SBATCH --output=queue_best_surfaces-%j.out
#SBATCH --error=queue_best_surfaces-%j.error
#SBATCH --constraint=haswell

# Load the input argument (i.e., the number of requested submissions).
# Defaults to 100 total submissions. Note that you should change `n_systems` manually
# if you add or subtract systems
n_submissions=${1:-100}
n_systems=3
# Calculate how many surfaces we should be submitted per system based on some rough
# ballpark figures. Feel free to change them.
sites_per_system=16
surfaces_per_system=$((n_submissions / n_systems / sites_per_system))

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh

# Use Luigi to queue various surfaces for simulation

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["CO", "H"]' \
    --performance-threshold 0.1 \
    --max-surfaces $surfaces_per_system \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["H"]' \
    --performance-threshold 0.1 \
    --max-surfaces $surfaces_per_system \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300
