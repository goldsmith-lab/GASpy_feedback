#!/bin/bash -l

# Load the input argument (i.e., the number of requested submissions).
# Defaults to 100 total submissions. Note that you should change `n_systems` manually
# if you add or subtract systems
n_submissions=${1:-100}
n_systems=12
n_workers=${2:-4}
# Calculate how many surfaces we should be submitted per system based on some rough
# ballpark figures. Feel free to change them.
sites_per_surface=4
surfaces_per_system=$((n_submissions / n_systems / sites_per_surface))

# Load GASpy environment and variables
. ~/GASpy/.load_env.sh

# Use Luigi to queue various surfaces for simulation

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H","CO"]' \
    --mpid-list '["mp-1008555"]' \
    --miller-list '[[1, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-3574"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-4771"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-1022721"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-1022721"]' \
    --miller-list '[[2, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-867306"]' \
    --miller-list '[[2, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-998"]' \
    --miller-list '[[1, 1, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12777"]' \
    --miller-list '[[1, 1, 1]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12802"]' \
    --miller-list '[[0, 0, 1]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300

PYTHONPATH=$PYTHONPATH luigi \
    --module gaspy_feedback.feedback Surfaces \
    --ads-list '["H", "CO"]' \
    --mpid-list '["mp-12777"]' \
    --miller-list '[[1, 0, 0]]' \
    --xc 'rpbe' \
    --max-surfaces $surfaces_per_sytem \
    --scheduler-host $LUIGI_PORT \
    --worksers=$n_workers \
    --parallel-scheduling \
    --log-level=WARNING \
    --worker-timeout 300
