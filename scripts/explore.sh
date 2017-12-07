#!/bin/sh -l

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
    --scheduler-host $luigi_port \
    --workers=1 \
    --log-level=WARNING \
    --worker-timeout 300 
