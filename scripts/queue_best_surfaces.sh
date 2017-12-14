#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=queue_best_surfaces
#SBATCH --output=queue_best_surfaces-%j.out
#SBATCH --error=queue_best_surfaces-%j.error
#SBATCH --constraint=haswell

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh

# Use Luigi to queue various surfaces for simulation

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["CO", "H"]' \
    --performance-threshold 0.1 \
    --max-surfaces 16 \
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
    --max-surfaces 16 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300
