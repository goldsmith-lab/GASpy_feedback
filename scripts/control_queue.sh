#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=regular
#SBATCH --job-name=control_queue
#SBATCH --output=control_queue-%j.out
#SBATCH --error=control_queue-%j.error
#SBATCH --constraint=haswell

# TODO:  This is pretty hacky. We should probably write this in PERL, not bash. And use PERL to iterate.

##### Defining feedback splits (percents) #####
# For explorations, predictions, best surfaces, and surface blaster, respectively
sp_exp=15
sp_pred=70
sp_best_surfs=15
sp_surfs=0

##### Read system variables #####
# Load GASpy environment and variables
. ~/GASpy/.load_env.sh
# The full folder that this file is in
path="$GASPY_PATH/GASpy_feedback/scripts"
# Input argument:  The time interval for control (in hrs/batch of submission). Defaults to 6.
interval=${1:-6}

##### Read dynamic variables #####
# Failure rates (in percentages) for exploration, predictions, best surfaces, and surface blaster, respectively
f_exp=$(cat $path/control_variables.txt | grep f_exp | grep -o -E '[-]?[0-9]+')
f_pred=$(cat $path/control_variables.txt | grep f_pred | grep -o -E '[-]?[0-9]+')
f_best_surfs=$(cat $path/control_variables.txt | grep f_best_surfs | grep -o -E '[-]?[0-9]+')
f_surfs=$(cat $path/control_variables.txt | grep f_surfs | grep -o -E '[-]?[0-9]+')
# Calculation rate (jobs/day)
calc_rate=$(cat $path/control_variables.txt | grep calc_rate | grep -o -E '[0-9]+')

##### Calculate control variables #####
# Desired length of the queue (jobs). We add a 100 % safety buffer
Qr=$((calc_rate / 24 * interval * 2))
# The number of things queued right now
Q=$(lpad -l $LPAD_PATH get_fws -d count -s READY)
# The negative error between what's queue and how many we want queued
err=$((Qr - Q))
# The number of jobs to try to submit for each feedback
n_exp=$((err * sp_exp / (100 - f_exp)))
n_pred=$((err * sp_pred / (100 - f_pred)))
n_best_surfs=$((err * sp_best_surfs / (100 - f_best_surfs)))
n_surfs=$((err * sp_surfs / (100 - f_surfs)))

##### Simultaneously implement control of queue length and update dynamic variables #####
# Explorations
if [ "$n_exp" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH get_fws -d count)
    bash $path/explore.sh $n_exp
    qf=$(lpad -l $LPAD_PATH get_fws -d count)
    dq=$((qf - q0))
    # Modify the failure rate
    f_exp=$((100 - 100 * dq / n_exp))
    if [ $f_exp -gt "90" ]; then  # To prevent singularities, we keep failure rates below 90
        f_exp=90
    fi
    sed -i "s/f_exp=.*/f_exp=$f_exp/g" $path/control_variables.txt
fi

# Predictions
if [ "$n_pred" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH get_fws -d count)
    bash $path/queue_predicted.sh $n_pred
    qf=$(lpad -l $LPAD_PATH get_fws -d count)
    dq=$((qf - q0))
    # Modify the failure rate
    f_pred=$((100 - 100 * dq / n_pred))
    if [ $f_pred -gt "90" ]; then  # To prevent singularities, we keep failure rates below 90
        f_pred=90
    fi
    sed -i "s/f_pred=.*/f_pred=$f_pred/g" $path/control_variables.txt
fi

# Best surfaces
if [ "$n_best_surfs" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH get_fws -d count)
    bash $path/queue_best_surfaces.sh $n_best_surfs
    qf=$(lpad -l $LPAD_PATH get_fws -d count)
    dq=$((qf - q0))
    # Modify the failure rate
    f_best_surfs=$((100 - 100 * dq / n_best_surfs))
    if [ $f_best_surfs -gt "90" ]; then  # To prevent singularities, we keep failure rates below 90
        f_best_surfs=90
    fi
    sed -i "s/f_best_surfs=.*/f_best_surfs=$f_best_surfs/g" $path/control_variables.txt
fi

# Surface blaster
if [ "$n_surfs" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH get_fws -d count)
    bash $path/queue_surfaces.sh $n_surfs
    qf=$(lpad -l $LPAD_PATH get_fws -d count)
    dq=$((qf - q0))
    # Modify the failure rate
    f_surfs=$((100 - 100 * dq / n_surfs))
    if [ $f_surfs -gt "90" ]; then  # To prevent singularities, we keep failure rates below 90
        f_surfs=90
    fi
    sed -i "s/f_surfs=.*/f_surfs=$f_surfs/g" $path/control_variables.txt
fi
