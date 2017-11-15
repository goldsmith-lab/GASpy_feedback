#!/bin/sh

module load python
cd /global/project/projectdirs/m2755/GASpy_dev/GASpy_feedback
source activate /project/projectdirs/m2755/GASpy_conda/

PYTHONPATH='.' luigi \
    --module feedback Surfaces \
    --ads-list '["CO","H","O","OH","C"]' \
    --mpid-list '["mp-30","mp-81","mp-124","mp-23","mp-2","mp-126","mp-74"]' \
    --miller-list '[[1,0,0],[1, 1, 1],[2,1,1]]' \
    --xc 'rpbe' \
    --max-submit 100 \
    --scheduler-host 128.55.224.51 \
    --workers=4 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
