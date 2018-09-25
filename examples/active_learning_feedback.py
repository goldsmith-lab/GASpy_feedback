'''
This is an example script is used to make FireWorks rockets---i.e., submit
calculations for---low-coverage adsorption sites whose predicted adsorption
energies are near a specified target. Sites are chosen with Gaussian noise.

We submit continuously to try and hit our target queue size. If we actually
hit our queue size, then pause for an hour before checking back for
more submissions.

Args:
    user                    A string indicating which FireWorks user you are.
                            This helps us manage separate queues within the
                            same LaunchPad. Defaults to your Linux user name.
    target_queue_size       The number of rockets you want either ready or
                            running for a given user.
    adsorbates_and_targets  A list of 2-tuples where the first element in
                            each tuple is a list of the [co]adsorbates you
                            want to bulid rockets for and the second element
                            is the adsorption energy you want to target in eV.
    model                   A string for the model that you want to use;
                            see the notebooks in GASpy_regressions
    stdev                   We select sites around the target using Gaussian
                            noise. This argument sets the standard deviation
                            of that Gaussian noise in eV.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import os
import time
import argparse
from gaspy.utils import get_lpad
from gaspy.tasks import run_tasks
from gaspy_feedback import BestLowCoverageSitesWithGaussianNoise


# Set defaults for arguments and create command-line parser for them
parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default=os.getlogin())
parser.add_argument('--target_queue_size', type=int, default=300)
parser.add_argument('--adsorbates_and_targets', type=list, default=[(['CO'], -0.67),
                                                                    (['O'], 2.403)])
parser.add_argument('--model', type=str, default='model0')
parser.add_argument('--stdev', type=float, default=0.1)

args = parser.parse_args()
user = args.user
target_queue_size = args.target_queue_size
adsorbates_and_targets = args.adsorbates_and_targets
model = args.model
stdev = args.stdev

# Make a function to count the number of rockets.
# This'll help us keep track of how many rockets to build.
lpad = get_lpad()
def get_num_rockets():  # noqa: E302
    '''
    Gets the number of rockets that are ready and/or running.

    Arg:
        user    A string for the user whose rockets you want to count
    '''
    num_ready = lpad.get_fw_ids({'name.user': user, 'state': 'READY'}, count_only=True)
    num_running = lpad.get_fw_ids({'name.user': user, 'state': 'RUNNING'}, count_only=True)
    num_rockets = num_ready + num_running
    return num_rockets


def build_rockets():
    '''
    Try to build as many rockets as are needed to hit our target queue size.
    '''
    num_rockets_to_build = target_queue_size - get_num_rockets()
    rockets_per_task = int(num_rockets_to_build / len(adsorbates_and_targets))

    if rockets_per_task > 0:
        tasks = []
        for adsorbates, target in adsorbates_and_targets:
            task = BestLowCoverageSitesWithGaussianNoise(adsorbates=adsorbates,
                                                         energy_target=target,
                                                         model_tag=model,
                                                         stdev=stdev,
                                                         max_rockets=rockets_per_task)
            tasks.append(task)

        n_rockets_before = get_num_rockets()
        run_tasks(tasks, workers=2)
        n_rockets_after = get_num_rockets()
        n_rockets_made = n_rockets_after - n_rockets_before
        print('%i rockets made successfully out of %i.' % (n_rockets_made, num_rockets_to_build))


# Run continuously with a 1 hour pause if we are actually above quota.
while True:
    if get_num_rockets() >= target_queue_size:
        time.sleep(3600)
    else:
        build_rockets()
