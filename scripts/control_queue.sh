#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --partition=regular
#SBATCH --job-name=control_queue
#SBATCH --output=control_queue-%j.out
#SBATCH --error=control_queue-%j.error
#SBATCH --constraint=haswell

# TODO:  This is pretty hacky. We should probably write this in PERL, not bash. And use PERL to iterate.

##### Defining feedback splits (percents) #####
# For explorations, predictions, best surfaces, and surface blaster, respectively
sp_exp=0
sp_pred=80
sp_best_surfs=20
sp_surfs=0

##### Read system variables #####
# The full folder that this file is in
path=$(cd -P "$(dirname "$_")" && pwd)
# Load GASpy environment and variables
. ~/GASpy/.load_env.sh
# Input argument:  The time interval for control (in hrs/batch of submission). Defaults to 3.
interval=${1:-3}

##### Read dynamic variables #####
# Failure rates (in percentages) for exploration, predictions, best surfaces, and surface blaster, respectively
f_exp=$(cat $path/control_variables.txt | grep f_exp | grep -o -E '[0-9]+')
f_pred=$(cat $path/control_variables.txt | grep f_pred | grep -o -E '[0-9]+')
f_best_surfs=$(cat $path/control_variables.txt | grep f_best_surfs | grep -o -E '[0-9]+')
f_surfs=$(cat $path/control_variables.txt | grep f_surfs | grep -o -E '[0-9]+')
# Calculation rate (jobs/day)
calc_rate=$(cat $path/control_variables.txt | grep calc_rate | grep -o -E '[0-9]+')

##### Calculate control variables #####
# Desired length of the queue (jobs). We add a 50% safety buffer
Qr=$((calc_rate / 24 * interval * 3 / 2))
# The number of things queued right now
Q=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
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
    q0=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    bash $path/explore.sh $n_exp
    qf=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    dq=$((qf - q0))
    # Modify the failure rate
    if [ "dq" -ne "0" ]; then  # Calculate the rate only if it's not 100% (to prevent singularities)
        f_exp=$((100 - 100 * dq / n_exp))
    else
        f_exp=90  # Approximate total failure with a high failure
    fi
    sed -i "s/f_exp=.*/f_exp=$f_exp/g" $path/control_variables.txt
fi
# Predictions
if [ "$n_pred" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    bash $path/queue_predicted.sh $n_pred
    qf=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    dq=$((qf - q0))
    # Modify the failure rate
    f_pred=$((100 - 100 * dq / n_pred))
    if [ "dq" -ne "0" ]; then  # Calculate the rate only if it's not 100% (to prevent singularities)
        f_exp=$((100 - 100 * dq / n_pred))
    else
        f_exp=90  # Approximate total failure with a high failure
    fi
    sed -i "s/f_pred=.*/f_pred=$f_pred/g" $path/control_variables.txt
fi
# Best surfaces
if [ "$n_best_surfs" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    bash $path/queue_best_surfaces.sh $n_best_surfs
    qf=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    dq=$((qf - q0))
    # Modify the failure rate
    f_best_surfs=$((100 - 100 * dq / n_best_surfs))
    if [ "dq" -ne "0" ]; then  # Calculate the rate only if it's not 100% (to prevent singularities)
        f_exp=$((100 - 100 * dq / n_best_surfs))
    else
        f_exp=90  # Approximate total failure with a high failure
    fi
    sed -i "s/f_best_surfs=.*/f_best_surfs=$f_best_surfs/g" $path/control_variables.txt
fi
# Surface blaster
if [ "$n_surfs" -gt "0" ]; then
    # Sumbit the jobs and calculate the number of jobs actually submitted, `dq`
    q0=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    bash $path/queue_surfaces.sh $n_surfs
    qf=$(lpad -l $LPAD_PATH report | grep READY | head -1 | grep -o -E '[0-9]+')
    dq=$((qf - q0))
    # Modify the failure rate
    f_surfs=$((100 - 100 * dq / n_surfs))
    if [ "dq" -ne "0" ]; then  # Calculate the rate only if it's not 100% (to prevent singularities)
        f_exp=$((100 - 100 * dq / n_surfs))
    else
        f_exp=90  # Approximate total failure with a high failure
    fi
    sed -i "s/f_surfs=.*/f_surfs=$f_surfs/g" $path/control_variables.txt
fi
