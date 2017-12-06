#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:30:00
#SBATCH --partition=regular
#SBATCH --job-name=explore
#SBATCH --output=explore-%j.out
#SBATCH --error=explore-%j.error
#SBATCH --constraint=haswell

# Go back to home directory, then go to GASpy
cd
cd GASpy/
# Get path information from the .gaspyrc.json file
conda_path="$(python .read_rc.py conda_path)"
luigi_port="$(python .read_rc.py luigi_port)"

# Load the appropriate environment, etc.
module load python
cd GASpy_feedback/gaspy_feedback
source activate $conda_path

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
