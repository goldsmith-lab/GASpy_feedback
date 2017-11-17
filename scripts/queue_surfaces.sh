#!/bin/sh

# Get path information from the .gaspyrc.json file
gaspy_path="$(python ../../.read_rc.py gaspy_path)"
conda_path="$(python ../../.read_rc.py conda_path)"
luigi_port="$(python ../../.read_rc.py luigi_port)"

# Load the appropriate environment, etc.
module load python
cd $gaspy_path/GASpy_feedback/gaspy_feedback
source activate $conda_path

# Use Luigi to queue various surfaces for simulation
PYTHONPATH=$PYTHONPATH luigi \
    --module feedback Surfaces \
    --ads-list '["CO","H","O","OH","C"]' \
    --mpid-list '["mp-30","mp-81","mp-124","mp-23","mp-2","mp-126","mp-74"]' \
    --miller-list '[[1,0,0],[1, 1, 1],[2,1,1]]' \
    --xc 'rpbe' \
    --max-submit 100 \
    --scheduler-host $luigi_port \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
