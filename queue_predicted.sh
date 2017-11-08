#!/bin/sh

module load python
cd /global/project/projectdirs/m2755/GASpy/GASpy_feedback
source activate /project/projectdirs/m2755/GASpy_conda/

PYTHONPATH='.' luigi \
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
    --scheduler-host 128.55.224.51 \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300 
