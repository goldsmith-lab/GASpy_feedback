#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=queue-%j.out
#SBATCH --error=queue-%j.error
#SBATCH --constraint=haswell

module load python
cd /global/project/projectdirs/m2755/GASpy_dev/GASpy_predict
source activate /project/projectdirs/m2755/GASpy_conda/

PYTHONPATH='.' luigi \
    --module feedback MatchingAdslabs \
    --xc 'rpbe' \
    --ads-list '["OOH"]' \
    --matching-ads 'OH' \
    --max-pred 300 \
    --scheduler-host gilgamesh.cheme.cmu.edu \
    --workers=16 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300
