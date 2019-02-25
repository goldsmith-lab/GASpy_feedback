'''
This is an example script is used to make FireWorks rockets---i.e., submit
calculations for---random adsorption sites in our catalog.

We submit continuously to try and hit our target queue size. If we actually hit
our queue size, then pause for an hour before checking back for more
submissions.

Args:
    user        A string indicating which FireWorks user you are. This helps
                us manage separate queues within the same LaunchPad. Defaults
                to your Linux user name.
    quota       The number of rockets you want either ready or running for a
                given user.
    adsorbate   A string indicating which adsorbate you want to do calculations
                for.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import os
import time
import argparse
from gaspy.tasks import schedule_tasks
from gaspy_feedback import get_n_jobs_to_submit, randomly


# Set defaults for arguments and create command-line parser for them
parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default=os.getlogin())
parser.add_argument('--adsorbate', type=str, default='CO')
parser.add_argument('--quota', type=int, default=300)
# Fetch the arguments
args = parser.parse_args()
user = args.user
adsorbate = args.adsorbate
quota = args.target_queue_size


def build_rockets():
    n_calcs = get_n_jobs_to_submit(user, quota)
    tasks = randomly(adsorbate, n_calcs=n_calcs)
    for task in tasks:
        schedule_tasks([task])


# Run continuously with a 1 hour pause if we are actually above quota.
while True:
    if get_n_jobs_to_submit(user, quota) > 0:
        time.sleep(3600)
    else:
        build_rockets()
