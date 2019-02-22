'''
This is an example script is used to make FireWorks rockets---i.e., submit
calculations for---low-coverage adsorption sites whose predicted 4 electron
onset potentials are near a specified target. Sites are chosen with Gaussian
noise.

We submit continuously to try and hit our target queue size. If we actually hit
our queue size, then pause for an hour before checking back for more
submissions.

Args:
    user        A string indicating which FireWorks user you are.  This helps
                us manage separate queues within the same LaunchPad. Defaults
                to your Linux user name.
    quota       The number of rockets you want either ready or running for a
                given user.
    adsorbate   A string indicating which adsorbate you want to do calculations
                for.
    target      A float indicating the onset potential you're targeting
    model       A string for the model that you want to use; see the notebooks
                in GASpy_regressions
    stdev       We select sites around the target using Gaussian noise. This
                argument sets the standard deviation of that Gaussian noise in
                eV.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import os
import time
import argparse
from gaspy.tasks import schedule_tasks
from gaspy_feedback import get_n_jobs_to_submit, orr_sites_with_gaussian_noise


# Set defaults for arguments and create command-line parser for them
parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default=os.getlogin())
parser.add_argument('--quota', type=int, default=300)
parser.add_argument('--adsorbate', type=str, default='OH')
parser.add_argument('--target', type=float, default=1.23)
parser.add_argument('--model', type=str, default='model0')
parser.add_argument('--stdev', type=float, default=0.2)
# Fetch the arguments
args = parser.parse_args()
user = args.user
quota = args.quota
adsorbate = args.adsorbate
target = args.target
model = args.model
stdev = args.stdev


def build_rockets():
    n_calcs = get_n_jobs_to_submit(user, quota)
    tasks = orr_sites_with_gaussian_noise(adsorbate=adsorbate,
                                          orr_target=target,
                                          stdev=stdev,
                                          n_calcs=n_calcs,
                                          model_tag=model)
    for task in tasks:
        schedule_tasks([task])


# Run continuously with a 1 hour pause if we are actually above quota.
while True:
    if get_n_jobs_to_submit(user, quota) > 0:
        time.sleep(3600)
    else:
        build_rockets()
