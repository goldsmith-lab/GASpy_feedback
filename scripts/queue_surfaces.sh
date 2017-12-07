#!/bin/sh -l

# Load GASpy
. ~/GASpy/scripts/load_env.sh
cd $GASPY_FB_PATH/gaspy_feedback

# Use Luigi to queue various surfaces for simulation

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H","CO"]' \
    --mpid-list '["mp-1008555"]' \
    --miller-list '[[1, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-3574"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-4771"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-1022721"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-1022721"]' \
    --miller-list '[[2, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[2, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-998"]' \
    --miller-list '[[1, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12777"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12802"]' \
    --miller-list '[[0, 0, 1]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12777"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces 1 \
    --scheduler-host $LUIGI_PORT \
    --workers=1 \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300
