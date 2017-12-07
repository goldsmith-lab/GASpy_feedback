#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=explore
#SBATCH --output=explore-%j.out
#SBATCH --error=explore-%j.error
#SBATCH --constraint=haswell

# Load GASpy
. ~/GASpy/scripts/load_env.sh
cd $GASPY_FB_PATH/gaspy_feedback

# Tell Luigi to queue various simulations based on a model's predictions
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Explorations \
    --ads-list '["CO", "H", "C", "O", "OH"]' \
    --fingerprints '["coordination"]' \
    --queries '["$processed_data.calculation_info.fp_init.coordination"]' \
    --xc 'rpbe' \
    --max-submit 200 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300 
