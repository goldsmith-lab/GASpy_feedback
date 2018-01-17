'''
These tasks tell GASpy what adsorption sites to simulated next.
There are different tasks for each type of selection style.

Here are some common task arguments in this module:
    xc          A string indicating the cross-correlational you want to use. As of now,
                it can only be `rpbe` or `beef-vdw`
    max_submit  A positive integer indicating the maximum number of simulations you want
                to queue up with one call of the task
    max_atoms   A positive integer indicating the maximum number of atoms you want
                present in the simulation
    ads_list    A list of strings, where the strings are the adsorbates you want to
                simulate
'''

__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'

import pdb  # noqa:  F401
import sys
import warnings
import luigi
from collections import OrderedDict
import pickle
import random
from gaspy import utils, defaults, gasdb
import gaspy_feedback.create_parameters as c_param

# Find the location of the GASpy tasks so that we can import/use them
configs = utils.read_rc()
tasks_path = configs['gaspy_path'] + '/gaspy'
sys.path.insert(0, tasks_path)
from tasks import FingerprintRelaxedAdslab, GenerateSlabs  # noqa: E402


# Load the default exchange correlational from the .gaspyrc.json file
XC = configs['default_xc']
MAX_SUBMIT = 20
MAX_ATOMS = 80


class RandomAdslabs(luigi.WrapperTask):
    ''' Simulated virtually random adsorption sites '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    max_submit = luigi.IntParameter(MAX_SUBMIT)
    max_atoms = luigi.IntParameter(MAX_ATOMS)

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
    Simulate adsorptions of one type of adsorbate based on what we've already simulated
    for a different type of adsorbate.

    Special input:
        matching_ads    A string indicating the adsorbate that "you've already simulated"
                        and want to copy sites for
    '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    matching_ads = luigi.Parameter()
    max_submit = luigi.IntParameter(MAX_SUBMIT)
    max_atoms = luigi.IntParameter(MAX_ATOMS)

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
    Use a fitted instance of `gaspy_regress.regressor.GASpyRegressor` to make predictions
    on the catalog, and then queue simulations that match a specified target.

    Special inputs:
        prediction_min          A float indicating a lower-bound for whatever property you are
                                predicting. Note that this may not be a "hard minimum" if you
                                choose particular priority methods, like 'gaussian'
        prediction_target       A float indicating the target you are trying to hit regarding
                                your predictions.
        prediction_max          A float indicating an upper-bound for whatever property you are
                                predicting. Note that this may not be a "hard maximum" if you
                                choose particular priority methods, like 'gaussian'
        predictions_location    A string indicating the location of the fitted instance of
                                GASpyRegressor that you are using
        block                   The block of the model that you are using to make predictions.
        priority                The way in which you want to prioritize the simulations.
                                Can probably be `targeted`, `random`, or `gaussian`.
                                See `gaspy_feedback.create_parameters._make_parameters_list`
                                for more details.
        n_sigmas                If priority == 'gaussian' or some other probability distribution
                                function, then it needs a standard deviation associated with it.
                                You may set it here. A higher value here yields a tighter targeting.
    '''
    # Set default values
    ads_list = luigi.ListParameter()
    prediction_min = luigi.FloatParameter(-4.)
    prediction_target = luigi.FloatParameter()
    prediction_max = luigi.FloatParameter(4.)
    predictions_location = luigi.Parameter()
    block = luigi.TupleParameter((None,))
    xc = luigi.Parameter(XC)
    max_submit = luigi.IntParameter(MAX_SUBMIT)
    priority = luigi.Parameter('gaussian')
    n_sigmas = luigi.FloatParameter(6.)
    max_atoms = luigi.IntParameter(MAX_ATOMS)

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
                                                       pkl=self.predictions_location,
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

    Special inputs:
        mpid_list       A list of strings indicating the mpid numbers you want to simulate
        miller_list     A list of list of integers indicating the miller indices you want
                        to simulate
        max_surfaces    A positive integer indicating the maximum number of surfaces you
                        want to queue simulations for. We use this in lieu of `max_submit`,
                        because this task submits surfaces naively. This means that we ask
                        for simulations on a surface without every knowing how many sites
                        (and therefore simulations) are on that surface. This is probably
                        dangerous, but we don't have time to think of a more elegant solution.
    '''
    # Set default values
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    mpid_list = luigi.ListParameter()
    miller_list = luigi.ListParameter()
    max_surfaces = luigi.IntParameter(MAX_SUBMIT)
    max_atoms = luigi.IntParameter(MAX_ATOMS)

    def requires(self):
        # Create parameters for every mpid/surface pairing
        parameters_list = []
        for mpid in self.mpid_list:
            for miller in self.miller_list:
                bulk = defaults.bulk_parameters(mpid, settings=self.xc)

                # Create the slabs if we have not yet done so
                GS = GenerateSlabs(parameters={'bulk': bulk,
                                               'slab': defaults.slab_parameters(miller, True, 0,
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

        # Submit for Fingerprinting/Relaxation. We shuffle the list to randomize what we run.
        random.shuffle(parameters_list)
        for i, parameters in enumerate(parameters_list):
            # This filter makes sure we don't accidentally submit too many jobs
            if i <= self.max_surfaces:
                parameters['adsorption']['numtosubmit'] = 100
                parameters['adsorption']['adsorbates'][0]['fp'] = {}
                yield FingerprintRelaxedAdslab(parameters=parameters)


class BestSurfaces(luigi.WrapperTask):
    '''
    This task will queue up and exhaust the adsorption sites of the highest performing
    surfaces that have not yet been simulated.

    Special inputs:
        predictions             A string indicating the location of a data ball
                                created by `gaspy_regress.predict`
        performance_threshold   A float (between 0 and 1, preferably) that indicates
                                the minimum level of performance relative to the best
                                performing surface.
        max_surfaces            A positive integer indicating the maximum number of surfaces you
                                want to queue simulations for. We use this in lieu of
                                `max_submit`, because this task submits surfaces naively. This
                                means that we ask for simulations on a surface without every
                                knowing how many sites (and therefore simulations) are on that
                                surface. This is probably dangerous, but we don't have time to
                                think of a more elegant solution.
    '''
    # Set default values
    predictions = luigi.Parameter()
    xc = luigi.Parameter(XC)
    ads_list = luigi.ListParameter()
    performance_threshold = luigi.FloatParameter(0.1)
    max_surfaces = luigi.IntParameter(MAX_SUBMIT)
    max_atoms = luigi.IntParameter(MAX_ATOMS)

    def requires(self):
        # Unpack the data structure
        with open(self.predictions, 'r') as f:
            data_ball = pickle.load(f)
        sim_data, unsim_data = data_ball
        sim_docs, predictions, _ = zip(*sim_data)
        cat_docs, estimations = zip(*unsim_data)
        _, y_data_pred = zip(*predictions)
        y_pred, y_u_pred = zip(*y_data_pred)
        _, y_data_est = zip(*estimations)
        y_est, y_u_est = zip(*y_data_est)
        # Package the estimations and predictions together, because we don't
        # really care which they come from. Then zip it up so we can sort everything
        # at once.
        docs = list(sim_docs)
        docs.extend(list(cat_docs))
        y = list(y_pred)
        y.extend(list(y_est))
        y_u = list(y_u_pred)
        y_u.extend(list(y_u_est))
        data = zip(docs, y, y_u)

        # Sort the data so that the items with the highest `y` values are
        # at the beginning of the list
        data = sorted(data, key=lambda datum: datum[1], reverse=True)
        # Take out everything that hasn't performed well enough
        y_max = data[0][1]
        data = [(doc, _y, _y_u) for doc, _y, _y_u in data
                if _y > self.performance_threshold*y_max]
        # Find the best performing surfaces
        best_surfaces = []
        for doc, _, _ in data:
            mpid = doc['mpid']
            miller = tuple(doc['miller'])
            surface = (mpid, miller)
            best_surfaces.append(surface)

        # Find all of the adsorption sites that we have not yet simulated
        docs = gasdb.unsimulated_catalog(adsorbates=self.ads_list,
                                         calc_settings=self.xc,
                                         max_atoms=self.max_atoms)
        # Figure out the surfaces that we have not yet simulated
        unsim_surfaces = []
        for doc in docs:
            mpid = doc['mpid']
            miller = tuple(doc['miller'])
            surface = (mpid, miller)
            unsim_surfaces.append(surface)
        # Eliminate redundancies in our list of unsimulated surfaces
        unsim_surfaces = list(set(unsim_surfaces))
        # Store the unsimulated surfaces into they keys of a dictionary so
        # that we can search through them quickly
        unsim_surfaces = dict.fromkeys(unsim_surfaces)

        # Queue up the top performing surfaces if they are unsimulated
        n = 0
        for surface in best_surfaces:
            # Stop if we've submitted enough
            if n >= self.max_surfaces:
                break
            # Otherwise, keep submitting
            if surface in unsim_surfaces:
                n += 1
                mpid, miller = surface
                yield Surfaces(xc=self.xc, ads_list=self.ads_list,
                               mpid_list=[mpid], miller_list=[miller],
                               max_surfaces=1, max_atoms=self.max_atoms)


class Explorations(luigi.WrapperTask):
    '''
    This task will queue up a random assortment of adsorption sites that have brand new
    properties that we have not yet simulated. It is meant to be used as a learning seed
    when adding new materials to the catalog.

    Special input:
        fingerprints    A sequence of strings indicating which property you want to explore.
                        If this string does not appear in `gaspy.defaults.fingerprints`,
                        then you need to specify the `queries` argument
        queries         A sequence of strings corresponding to `fingerprints` that contain
                        the appropriate mongo/GASdb queries to find the fingerprints.
                        This is only needed if there are any fingerprints that are not
                        in the default set of fingerprints. The length of this sequence
                        must be equal to the length of `fingerprints`. Any queries
                        provided for default fingerprints will be ignored.
    '''
    # Set default values. And define arguments.
    ads_list = luigi.ListParameter()
    xc = luigi.Parameter(XC)
    max_submit = luigi.IntParameter(MAX_SUBMIT)
    max_atoms = luigi.IntParameter(MAX_ATOMS)
    fingerprints = luigi.ListParameter()
    queries = luigi.ListParameter()

    def requires(self):
        # Set the fingerprints that we use to pull/create the mongo documents
        fingerprints = defaults.fingerprints()
        for i, fp in enumerate(self.fingerprints):
            if fp not in fingerprints:
                fingerprints[fp] = self.queries[i]
        # Calculate the number of submissions we're allowed to do per adsorbate
        submit_per_ads = self.max_submit/len(self.ads_list)

        # Pull out everything that we have and have not simulated so far
        for ads in self.ads_list:
            sim_docs = gasdb.get_docs(adsorbates=[ads],
                                      calc_settings=self.xc,
                                      fingerprints=fingerprints)
            unsim_docs = gasdb.unsimulated_catalog([ads],
                                                   calc_settings=self.xc,
                                                   fingerprints=fingerprints)

            # Make a function to pull out a "tag", which is a [possibly nested] tuple
            # of the values of the fingerprints that matter.
            def pull_tag(doc):
                tag = []
                for fp in self.fingerprints:
                    try:
                        _tag = doc[fp]
                        if isinstance(_tag, list):
                            _tag = tuple(_tag)
                        tag.append(_tag)
                    except KeyError:
                        warnings.warn('The "%s" fingerprint was not found in the following document; skipping this doc\n%s')
                return tuple(tag)

            # Figure out what "tags" we've already explored, where a "tag"
            # is a tuple of the values of the fingerprints that matter.
            # Note that we turn `explored_tags` into a dictionary for fast lookup
            explored_tags = []
            for doc in sim_docs:
                explored_tags.append(pull_tag(doc))
            explored_tags = dict.fromkeys(explored_tags)
            # Filter out items that we have already simulated
            docs_to_explore = []
            for doc in unsim_docs:
                unsim_tag = pull_tag(doc)
                if unsim_tag not in explored_tags:
                    docs_to_explore.append(doc)
            print('Still have %i sites left to explore' % (len(docs_to_explore)))

            # Queue up the things that we want to explore in a randomized order
            random.seed(42)
            parameters_list = c_param._make_parameters_list(docs_to_explore, ads,
                                                            prioritization='CoordinationLength',
                                                            max_predictions=submit_per_ads)
            for parameters in parameters_list:
                yield FingerprintRelaxedAdslab(parameters=parameters)
