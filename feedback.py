'''
These tasks use the `create_parameters` module to determine what systems
to simulate next, and then it submits the simulations. There are different
tasks for each type of selection style.

For more information regarding all of the task arguments, please
refer to the respective functions that each tasks calls from `create_parameters`.
'''
# pylint: disable=unsubscriptable-object

__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'

import pdb  # noqa:  F401
import sys
import luigi
import create_parameters as c_param
sys.path.insert(0, '../')
from tasks import FingerprintRelaxedAdslab  # noqa:  E401


# XC = 'beef-vdw'
XC = 'rpbe'


class RandomAdslabs(luigi.WrapperTask):
    '''
    Use the `create_parameters.randomly` function to create a list of GASpy
    parameters, then submit them for simulation
    '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    max_submit = luigi.IntParameter(20)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:       # pylint: disable=not-an-iterable
            parameters_list = c_param.randomly(ads, calc_settings=self.xc,
                                               max_predictions=submit_per_ads)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class MatchingAdslabs(luigi.WrapperTask):
    '''
    Use the `create_parameters.from_matching_ads` function to create a list of GASpy
    parameters, then submit them for simulation.

    Note that `matching_ads` is a string indicating the adsorbate you want to "copy",
    while `ads_list` is a list of the adsorbates that you want to make submissions for.
    '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    matching_ads = luigi.Parameter()
    max_submit = luigi.IntParameter(20)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:       # pylint: disable=not-an-iterable
            parameters_list = c_param.from_matching_ads(ads, self.matching_ads,
                                                        calc_settings=self.xc,
                                                        max_predictions=submit_per_ads)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class Predictions(luigi.WrapperTask):
    '''
    Use the `create_parameters.from_predictions` function to create a list of GASpy
    parameters, then submit them for simulation
    '''
    # Set default values
    ads_list = luigi.ListParameter()
    prediction_min = luigi.FloatParameter(-4.)
    prediction_target = luigi.FloatParameter()
    prediction_max = luigi.FloatParameter(4.)
    model_location = luigi.Parameter()
    block = luigi.Parameter('no_block')
    xc = luigi.Parameter(XC)
    max_submit = luigi.IntParameter(20)
    priority = luigi.Parameter('gaussian')
    n_sigmas = luigi.FloatParameter(6.)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:   # pylint: disable=not-an-iterable
            parameters_list = c_param.from_predictions(ads,
                                                       self.prediction_min,
                                                       self.prediction_target,
                                                       self.prediction_max,
                                                       pkl=self.model_location,
                                                       block=self.block,
                                                       calc_settings=self.xc,
                                                       max_predictions=submit_per_ads,
                                                       prioritization=self.priority,
                                                       n_sigmas=self.n_sigmas)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)
