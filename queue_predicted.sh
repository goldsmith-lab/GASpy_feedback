#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=2:00:00
#SBATCH --partition=regular
#SBATCH --job-name=vasp
#SBATCH --output=queuepredicted-%j.out
#SBATCH --error=queuepredicted-%j.error
#SBATCH --constraint=haswell

cd /global/project/projectdirs/m2755/GASpy/GASpy_predict

PYTHONPATH='.' luigi \
    --module feedback CoordcountNncToEnergy \
    --xc 'rpbe' \
    --ads-list '["CO"]' \
    --energy-target -0.55 \
    --max-pred 50 \
    --scheduler-host gilgamesh.cheme.cmu.edu  \
    --workers=8 \
    --log-level=WARNING \
    --parallel-scheduling \
    --worker-timeout 300 
