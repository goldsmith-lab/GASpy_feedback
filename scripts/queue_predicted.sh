#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=queue_predicted
#SBATCH --output=queue_predicted-%j.out
#SBATCH --error=queue_predicted-%j.error
#SBATCH --constraint=haswell

# Load GASpy
. ~/GASpy/scripts/load_env.sh
cd $GASPY_FB_PATH/gaspy_feedback

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
