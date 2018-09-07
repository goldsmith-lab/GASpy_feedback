'''
This is an example script is used to make FireWorks rockets---i.e., submit
calculations for---low-coverage adsorption sites whose predicted adsorption
energies are near a specified target. Sites are chosen with Gaussian noise.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

from gaspy.tasks import run_tasks
from gaspy_feedback import BestLowCoverageSitesWithGaussianNoise


co_feedback_task = BestLowCoverageSitesWithGaussianNoise(adsorbates=['CO'],
                                                         energy_target=-0.67,
                                                         model_tag='model0',
                                                         stdev=0.1,
                                                         max_rockets=100)
h_feedback_task = BestLowCoverageSitesWithGaussianNoise(adsorbates=['H'],
                                                        energy_target=-0.27,
                                                        model_tag='model0',
                                                        stdev=0.1,
                                                        max_rockets=100)

tasks = [co_feedback_task, h_feedback_task]
run_tasks(tasks)
