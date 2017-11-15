'''
These tasks use the `create_parameters` module to determine what systems
to simulate next, and then it submits the simulations. There are different
tasks for each type of selection style.

For more information regarding all of the task arguments, please
refer to the respective functions that each tasks calls from `create_parameters`.
'''

__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'

import pdb  # noqa:  F401
import sys
import luigi
import create_parameters as c_param
from collections import OrderedDict
import pickle
from random import shuffle
import gaspy.defaults as defaults
sys.path.insert(0, '../')
from tasks import FingerprintRelaxedAdslab,MatchCatalogShift,GenerateSlabs  # noqa:  E401


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
    max_atoms = luigi.IntParameter(50)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:
            parameters_list = c_param.randomly(ads, calc_settings=self.xc,
                                               max_predictions=submit_per_ads,
                                               max_atoms=self.max_atoms)
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
    max_atoms = luigi.IntParameter(50)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:
            parameters_list = c_param.from_matching_ads(ads, self.matching_ads,
                                                        calc_settings=self.xc,
                                                        max_predictions=submit_per_ads,
                                                        max_atoms=self.max_atoms)
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
    block = luigi.TupleParameter('no_block')
    xc = luigi.Parameter(XC)
    max_submit = luigi.IntParameter(20)
    priority = luigi.Parameter('gaussian')
    n_sigmas = luigi.FloatParameter(6.)
    max_atoms = luigi.IntParameter(50)

    def requires(self):
        ''' This is a Luigi wrapper task, so it'll just do anything in this `requires` method '''
        # Find out how many adsorbates the user wants to be simulating. This helps us figure
        # out how many parameters we should be pulling out for each adsorbate.
        n_ads = len(self.ads_list)
        submit_per_ads = self.max_submit / n_ads

        for ads in self.ads_list:
            parameters_list = c_param.from_predictions(ads,
                                                       self.prediction_min,
                                                       self.prediction_target,
                                                       self.prediction_max,
                                                       pkl=self.model_location,
                                                       block=self.block,
                                                       calc_settings=self.xc,
                                                       max_predictions=submit_per_ads,
                                                       prioritization=self.priority,
                                                       n_sigmas=self.n_sigmas,
                                                       max_atoms=self.max_atoms)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class Surfaces(luigi.WrapperTask):
    '''
    Use the `create_parameters.randomly` function to create a list of all
    asorption sites on a set of surfaces, then submit them for simulation. Good for
    fully evaluating possibly interesting surfaces.
    '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    mpid_list = luigi.ListParameter()
    miller_list = luigi.ListParameter()
    max_submit = luigi.IntParameter(20)
    max_atoms = luigi.IntParameter(50)

    def requires(self):
        # Create parameters for every mpid/surface pairing
        parameters_list = []
        for mpid in self.mpid_list:
            for miller in self.miller_list:
                bulk = defaults.bulk_parameters(mpid, settings=self.xc)

                # Create the slabs if we have not yet done so
                GS = GenerateSlabs(parameters={'bulk':bulk,
                                               'slab':defaults.slab_parameters(miller, True, 0,
                                                                               settings=self.xc)})
                if not(GS.complete()):
                    yield GS

                # If we have the slabs, then create the parameters manually and submit
                else:
                    # Pull out the slabs and make parameters from them
                    slab_list = pickle.load(GS.output().open())
                    slabs = [slab for slab in slab_list if slab['tags']['miller'] == miller]
                    for slab in slabs:
                        slab = defaults.slab_parameters(miller,
                                                        slab['tags']['top'],
                                                        slab['tags']['shift'],
                                                        settings=self.xc)
                        # Fetch the adsorbates from the input and make parameters from them
                        for ads in self.ads_list:
                            gas = defaults.gas_parameters(ads, settings=self.xc)
                            adsorption = defaults.adsorption_parameters(ads, settings=self.xc)
                            # Combine all the parameters
                            parameters_list.append(OrderedDict(bulk=bulk,
                                                               slab=slab,
                                                               adsorption=adsorption,
                                                               gas=gas))

        # Submit for Fingerprinting/Relaxation. We shuffle the list to randomize what we run. This
        # really only matters if len(parameter_list) > max_submit
        shuffle(parameters_list)
        for i, parameters in enumerate(parameters_list):
            # This filter makes sure we don't accidentally submit too many jobs
            if i <= self.max_submit:
                parameters['adsorption']['numtosubmit'] = 100
                parameters['adsorption']['adsorbates'][0]['fp'] = {}
                yield FingerprintRelaxedAdslab(parameters=parameters)
