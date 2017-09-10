'''
This script contains classes/tasks for Luigi to execute.

These tasks use the `GASPredict` class in this submodule to determine what systems
to relax next, and then it performs the relaxation.

Here is a list of a lot of the inputs used in the various tasks:
    xc              Exchange correlational
    ads_list        The adsorbate(s) you want to look at
    model_location  The location of the pickled model we want to use for the feedback
                    loop. Include the name of the file, as well.
    priority        The prioritization (i.e., method of choosing the next relaxation).
                    Can be:
                        anything
                        targeted
                        random
                        gaussian
    energy_min      The minimum (predicted) adsorption energy of systems you want to simulate
    energy_max      The maximum (predicted) adsorption energy of systems you want to simulate
    energy_target   The target adsorption energy you want to simulate
    max_pred        The maximum number of predictions that we want to sen through the
                    feedback loop. Note that the total number of submissions will still
                    be MAX_PRED*len(ads_list)
'''
__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'
# Since this is in a submodule, we add the parent folder to the python path
import pdb
import sys
from gas_predict import GASPredict
sys.path.insert(0, '../')
from tasks import FingerprintRelaxedAdslab
import luigi


#XC = 'beef-vdw'
XC = 'rpbe'
MODEL_LOC = ''


class CoordcountAdsToEnergy(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above (or below).

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and overwriting the old model pickle with the new one).
    '''
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    model_location = luigi.Parameter(MODEL_LOC)
    priority = luigi.Parameter('gaussian')
    energy_min = luigi.Parameter(-4)
    energy_max = luigi.Parameter(4)
    energ_target = luigi.Parameter(-0.55)
    max_pred = luigi.IntParameter(10)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate
        for ads in self.ads_list:
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl=self.model_location,
                                     calc_settings=self.xc)
            parameters_list = gas_predict.energy_fr_coordcount_ads(self.energy_min,
                                                                   self.energy_max,
                                                                   self.energy_target,
                                                                   max_predictions=self.max_pred,
                                                                   prioritization=self.priority)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class CoordcountNncToEnergy(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above (or below).

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and overwriting the old model pickle with the new one).
    '''
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    model_location = luigi.Parameter(MODEL_LOC)
    priority = luigi.Parameter('gaussian')
    energy_min = luigi.FloatParameter(-4.)
    energy_max = luigi.FloatParameter(4.)
    energy_target = luigi.FloatParameter(-0.55)
    max_pred = luigi.IntParameter(10)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate
        for ads in self.ads_list:
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl=self.model_location,
                                     calc_settings=self.xc,
                                     block=(self.ads_list[0],))
            parameters_list = gas_predict.energy_fr_coordcount_nnc(self.energy_min,
                                                                   self.energy_max,
                                                                   self.energy_target,
                                                                   max_predictions=self.max_pred,
                                                                   prioritization=self.priority)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class MatchingAdslabs(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above.

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and re-pickling the new model).
    '''
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    matching_ads = luigi.Parameter()
    max_pred = luigi.IntParameter(10)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate (i.e., if we MAX_PRED = 50
        # and len(ads_list) == 2, then we will have (50*2=) 100 predictions. I made it this way
        # because I'm too lazy to code it better.
        for ads in self.ads_list:
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl='',
                                     calc_settings=self.xc)
            parameters_list = gas_predict.matching_ads(max_predictions=self.max_pred,
                                                       adsorbate=self.matching_ads)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class RandomAdslabs(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above.

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and re-pickling the new model).
    '''
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    max_pred = luigi.IntParameter(10)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate (i.e., if we MAX_PRED = 50
        # and len(ads_list) == 2, then we will have (50*2=) 100 predictions. I made it this way
        # because I'm too lazy to code it better.
        for ads in self.ads_list:
            fingerprints = {'coordination': '$processed_data.fp_init.coordination'}
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl='',
                                     calc_settings=self.xc,
                                     fingerprints=fingerprints)
            parameters_list = gas_predict.anything(max_predictions=self.max_pred)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)
