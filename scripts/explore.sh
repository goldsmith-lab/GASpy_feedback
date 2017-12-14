#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --job-name=explore
#SBATCH --output=explore-%j.out
#SBATCH --error=explore-%j.error
#SBATCH --constraint=haswell

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh

# Tell Luigi to queue various simulations based on a model's predictions
PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Explorations \
    --ads-list '["CO", "H", "C", "O", "OH"]' \
    --fingerprints '["coordination"]' \
    --queries '["$processed_data.calculation_info.fp_init.coordination"]' \
    --xc 'rpbe' \
    --max-submit 400 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300 
