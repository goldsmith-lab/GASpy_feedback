#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=queuepredicted-%j.out
#SBATCH --error=queuepredicted-%j.error
#SBATCH --constraint=haswell

module load python
cd /global/project/projectdirs/m2755/GASpy/GASpy_feedback
source activate /project/projectdirs/m2755/GASpy_conda/

PYTHONPATH='.' luigi \
    --module feedback Predictions \
    --ads-list '["CO"]' \
    --prediction-target -0.55 \
    --model-location '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/GP_around_TPOT_FEATURES_coordcount_ads_neighbors_coordcounts_RESPONSES_energy_BLOCKS_.pkl' \
    --priority 'gaussian' \
    --xc 'rpbe' \
    --max-submit 200 \
    --scheduler-host 128.55.224.41 \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300 
