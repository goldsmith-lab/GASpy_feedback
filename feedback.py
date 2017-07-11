'''
This script contains classes/tasks for Luigi to execute.

These tasks use the `GASPredict` class in this submodule to determine what systems
to relax next, and then it performs the relaxation.

Here is a list of this script's inputs:
    DB_LOC      Location of the Local databases. Do not include database names.
    XC          Exchange correlational
    ADS         The adsorbate(s) you want to look at
    MAX_DUMP    The maximum number of rows to dump from the AuxDB to the Local DBs.
                A setting of 0 means no limit. This is used primarily for testing.
    WRITE_DB    Whether or not we want to dump the Aux DB to the Local energies DB.
                ...used for testing?
    MODEL_LOC   The location of the pickled model we want to use for the feedback
                loop. Include the name of the file, as well.
    PRIORITY    The prioritization (i.e., method of choosing the next relaxation).
    MAX_PRED    The maximum number of predictions that we want to sen through the
                feedback loop. Note that the total number of submissions will still
                be MAX_PRED*len(ADS)
'''
__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'
# Since this is in a submodule, we add the parent folder to the python path
import pdb
import sys
from gas_predict import GASPredict
sys.path.insert(0, '../')
from tasks import UpdateAllDB
from tasks import FingerprintRelaxedAdslab
import luigi


DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB'    # Cori
#DB_LOC = '/Users/Ktran/Nerd/GASpy'                  # Local
XC = 'rpbe'
#XC = 'beef-vdw'
ADS = ['OOH']
MAX_DUMP = 0
WRITE_DB = True
MODEL_LOC = '/global/project/projectdirs/m2755/GASpy/GASpy_regressions/pkls/CoordcountAds_Energy_GP.pkl'
PRIORITY = 'anything'
MAX_PRED = 10


class CoordcountAdsToEnergy(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above.

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and re-pickling the new model).
    '''
    xc = luigi.Parameter(XC)
    max_processes = luigi.IntParameter(MAX_DUMP)
    write_db = luigi.BoolParameter(WRITE_DB)
    model_location = luigi.Parameter(MODEL_LOC)
    max_pred = luigi.IntParameter(MAX_PRED)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate
        for ads in ADS:
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl=self.model_location,
                                     calc_settings=self.xc)
            parameters_list = getattr(gas_predict, PRIORITY)(max_predictions=self.max_pred)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)


class RelaxedAdslabs(luigi.WrapperTask):
    '''
    Before Luigi does anything, let's declare this task's arguments and establish defaults.
    These may be overridden on the command line when calling Luigi. If not, we pull them from
    above.

    Note that the pickled models should probably be updated as well. But instead of re-running
    the regression here, our current setup has use using Cron to periodically re-run the
    regression (and re-pickling the new model).
    '''
    xc = luigi.Parameter(XC)
    max_processes = luigi.IntParameter(MAX_DUMP)
    write_db = luigi.BoolParameter(WRITE_DB)
    model_location = luigi.Parameter(MODEL_LOC)
    max_pred = luigi.IntParameter(MAX_PRED)

    def requires(self):
        '''
        Here, we use the GASPredict class to identify the list of parameters that we can use
        to run the next set of relaxations.
        '''
        # We need to create a new instance of the gas_predictor for each adsorbate. Thus,
        # max_predictions is actually max_predictions_per_adsorbate
        for ads in ADS:
            gas_predict = GASPredict(adsorbate=ads,
                                     pkl=self.model_location,
                                     calc_settings=self.xc)
            parameters_list = getattr(gas_predict, 'matching_ads')(max_predictions=self.max_pred,
                                                                   adsorbate='OH')
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)
