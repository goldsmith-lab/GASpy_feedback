#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --partition=regular
#SBATCH --job-name=queue_best_surfaces
#SBATCH --output=queue_best_surfaces-%j.out
#SBATCH --error=queue_best_surfaces-%j.error
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

# Use Luigi to queue various surfaces for simulation
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CO2RR_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["CO"]' \
    --performance-threshold 0.1 \
    --max-surfaces 8 \
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback BestSurfaces \
    --predictions '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/HER_predictions_GP_around_TPOT_FEATURES_coordcount_neighbors_coordcounts_RESPONSES_energy_BLOCKS_adsorbate.pkl' \
    --xc 'rpbe' \
    --ads-list '["H"]' \
    --performance-threshold 0.1 \
    --max-surfaces 8 \
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300
