'''
When given the location for a pickled GASpy_regressions model and information about what
kind of calculations the user wants to run (e.g., which adsorbate, which calculation settings,
how many runs to do next, etc.), this class will use the model to create a list of GASpy
`parameters` to run next.
'''
__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import pdb
import sys
import copy
import random
from collections import OrderedDict
import itertools
from pprint import pprint
import numpy as np
import scipy as sp
import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
sys.path.insert(0, '../')
from gaspy import defaults
from gaspy import utils
sys.path.insert(0, '../GASpy_regressions')


class GASPredict(object):
    def __init__(self, adsorbate,
                 pkl=None, features=None, block='no_block',
                 calc_settings='rpbe', fingerprints=None):
        '''
        Here, we open the pickle. And then depending on the model type, we assign different
        methods to use for predicting data. For reference:  These models are created by
        `regress.ipynb` in the `GASpy_regressions` submodule.

        We also initialize the `self.site_docs` object, which will be a list of dictionaries
        containing calculation information for cataloged sites that have not yet been
        relaxed/put into the adsorption database.

        All of our __init__ arguments (excluding the pkl) are used to both define how we
        want to create our `parameters_list` objects and how we filter the `self.site_docs` so
        that we don't try to re-submit calculations that we've already done.

        Input:
            adsorbate       A string indicating the adsorbate that you want to make a
                            prediction for.
            pkl             The location of a pickle dictionary object:
                                'model' key should contain the model object.
                                'pre_processors' key should contain a dictionary whose keys
                                 are the model features (i.e., the mongo doc attributes) and
                                 whose values are the already-fitted pre-processors
                                 (e.g., LabelBinarizer)
                            Note that the user may provide '' for this argument, as well. This
                            should be done if the user wants to call the "anything" method.
            features        A list of strings, where each string corresponds to the
                            feature that should be fed into the model. Order matters, because
                            the pre-processed features will be stacked in order of appearance
                            in this list.
            block           The block of the model that we should be using. See
                            GASpy_regression's `regress.ipynb` for more details regarding
                            what a `block` could be.
            calc_settings   The calculation settings that we want to use. If we are using
                            something other than beef-vdw or rpbe, then we need to do some
                            more hard-coding here so that we know what in the local energy
                            database can work as a flag for this new calculation method.
            fingerprints    A dictionary of fingerprints and their locations in our
                            mongo documents. This is how we can pull out more (or less)
                            information from our database.
        '''
        # Default value for fingerprints. Since it's a mutable dictionary, we define it
        # down here instead of in the __init__ line.
        if not fingerprints:
            fingerprints = defaults.fingerprints()

        # Save some arguments to the object for later use
        self.adsorbate = adsorbate
        self.calc_settings = calc_settings

        # Fetch mongo docs for our adsorption and catalog databases so that we can
        # start parsing out cataloged sites that we've already relaxed.
        with utils.get_adsorption_db() as ads_client:
            ads_docs = utils.get_docs(ads_client, 'adsorption',
                                      calc_settings=calc_settings,
                                      fingerprints=fingerprints,
                                      adsorbates=[self.adsorbate])[0]
        with utils.get_catalog_db() as cat_client:
            cat_docs = utils.get_docs(cat_client, 'catalog', fingerprints=fingerprints)[0]
        # Hash the docs so that we can filter out any items in the catalog
        # that we have already relaxed. Note that we keep `ads_hash` in a dict so
        # that we can search through it, but we turn `cat_hash` into a list so that
        # we can iterate through it alongside `cat_docs`
        ads_hash = self._hash_docs(ads_docs)
        cat_hash = list(self._hash_docs(cat_docs).keys())
        # Perform the filtering while simultaneously creating `site_docs`
        self.site_docs = [doc for i, doc in enumerate(cat_docs)
                          if cat_hash[i] not in ads_hash]

        # Create `ads_docs` by finding all of the entries. Do so by not specifying
        # an adsorbate and by adding the `adsorbate` key to the fingerprint (and
        # thus the doc, as well).
        ads_fp = defaults.fingerprints()
        ads_fp['adsorbates'] = '$processed_data.calculation_info.adsorbate_names'
        self.ads_docs = utils.get_docs(ads_client, 'adsorption',
                                       calc_settings=calc_settings,
                                       fingerprints=ads_fp)[0]

        # We put a conditional before we start working with the pickled model. This allows the
        # user to go straight for the `anything` method without needing to find a dummy model.
        if pkl:
            # Unpack the pickled model
            with open(pkl, 'rb') as f:
                pkl = pickle.load(f)
            try:
                self.model = pkl['model'][block]
            except KeyError:
                raise Exception('The block %s is not in the model %s' %(block, pkl))
            self.pre_processors = pkl['pp']
            norm = pkl['norm']
            predictor = self._predictor()

            # Pre-process, stack, and normalize the fingerprints into features so that
            # they may be passed as arguments to the model
            p_data = []
            for feature in features:
                p_data.append(self._preprocess_feature(feature))
            model_inputs = np.array([np.hstack(_input) for _input in zip(*p_data)])/norm

            # Predict the adsorption energies of our docs
            self.energies = predictor(model_inputs)

        else:
            print('No model provided. Only the "anything" or "matching_ads" methods will work.')


    def _hash_docs(self, docs):
        '''
        This method helps convert the important characteristics of our systems into hashes
        so that we may sort through them more quickly. This is important to do when trying to
        compare entries in our two databases; it helps speed things up. Check out the __init__
        method for more details.

        Note that this method will consume whatever iterator it is given.

        Input:
            docs    A mongo doc object that has been created using `_get_docs`; the unparsed
                    version that is a list of dictionaries.
        Output:
            systems     An ordered dictionary whose keys are hashes of the each doc in
                        `docs` and whose values are empty. This dictionary is intended
                        to be parsed alongside another `docs` object, which is why
                        it's ordered.
        '''
        systems = OrderedDict()
        for doc in docs:
            # `system` will be one long string of the fingerprints
            system = ''
            for key, value in doc.iteritems():
                # Note that we turn the values into strings explicitly, because some
                # fingerprint features may not be strings (e.g., list of miller indices).
                system += str(key + '=' + str(value) + '; ')
            systems[hash(system)] = None

        return systems


    def _predictor(self):
        '''
        Each model type requires different methods/calls in order to be able to use
        them to make predictions. To make model usage easier, this method will return
        a function with a standardized input and standardized output. Below are the
        standardized inputs and outputs.

        Note that some models' inputs may deviate from this standard input. Refer
        to the specific functions' docstrings for details.

        Input:
            inputs      A list of pre-processed factors/inputs that the model can accept
        Output:
            responses   A list of values that the model predicts for the docs
        '''
        def sk_predict(inputs):
            ''' Assuming that the model in the pickle was an SKLearn model '''
            # SK models come with a method, `predict`, to transform the pre-processed input
            # to the output. Let's use it.
            return self.model.predict(inputs)

        # TODO:  Test this method
        def ala_predict(inputs):
            ''' Assuming that the model in the pickle was an alamopy model '''
            # Alamopy models come with a key, `f(model)`, that yields a lambda function to
            # transform the pre-processed input to the output. Let's use it.
            return self.model['f(model)'](inputs)

        # TODO:  Test this method
        def hierarch_predict(inputs_outer, inputs_inner):
            '''
            Assuming that the model in the pickle is a function created by GASpy_regression's
            `RegressionProcessor.hierarchical` method.

            Inputs:
                inputs_outer    A list of pre-processed factors/inputs that the outer model can accept.
                inputs_inner    A list of pre-processed factors/inputs that the outer model can accept.
            '''
            return self.model(inputs_outer, inputs_inner)

        # Figure out the model type and assign the correct function to perform predictions
        if isinstance(self.model, type(LinearRegression())):
            predictor = sk_predict
        elif isinstance(self.model, type(GradientBoostingRegressor())):
            predictor = sk_predict
        elif isinstance(self.model, type(GaussianProcessRegressor())):
            predictor = sk_predict
        elif isinstance(self.model, Pipeline):
            predictor = sk_predict
        elif isinstance(self.model, dict):
            predictor = ala_predict
        elif callable(self.model):
            predictor = hierarch_predict
        else:
            raise Exception('We have not yet established how to deal with this type of model')

        return predictor


    def _preprocess_feature(self, feature):
        '''
        This method will pre-process whatever data the user wants. This helps turn
        "raw" data into numerical data that is accepted by models.

        Input:
            feature A string that represents the type of feature that
                    you want to pull out of the fingerprints. See the
                    `GASpy_regressions` repository for more details.
        Output:
            p_data  A numpy array of pre-processed data. Each row represents
                    a new site, and each column represents a new numerical value
                    (depending on whatever the feature is).
        '''
        docs = self.site_docs

        if feature == 'coordcount':
            # lb = label binarizer (to turn the coordination into an array of sparse
            # binaries)
            lb = self.pre_processors['coordination']
            p_data = np.array([np.sum(lb.transform(coord.split('-')), axis=0)
                               for coord in [doc['coordination'] for doc in docs]])

        elif feature == 'ads':
            # lb = label binarizer (to turn the adsorbate into a vector of sparse
            # binaries)
            lb = self.pre_processors['adsorbate']
            p_data = lb.transform([self.adsorbate])[0]

        elif feature == 'nnc_count':
            # lb = label binarizer (to turn the nextnearestcoordination into an
            # array of sparse binaries)
            lb = self.pre_processors['nextnearestcoordination']
            p_data = np.array([np.sum(lb.transform(coord.split('-')), axis=0)
                               for coord in [doc['nextnearestcoordination'] for doc in docs]])

        else:
            raise Exception('That type of feature is not recognized')

        return p_data


    def _make_parameters_list(self, docs, prioritization, max_predictions,
                              target=None, values=None, n_sigmas=6):
        # pylint: disable=too-many-statements, too-many-branches
        '''
        Given the remaining mongo doc objects, this method will decide which of those
        docs to return for further relaxations. We do this in two steps:  1) choose and
        use a prioritization method (i.e., how to pick the docs), and then 2) trim the
        docs down to the number of predictions we want.

        Inputs:
            docs            A list of mongo doc objects
            prioritization  A string corresponding to a particular prioritization method.
                            So far, valid values include:
                                targeted (try to hit a single value, `target`)
                                random (randomly chosen)
                                gaussian (gaussian spread around target)
            max_predictions A maximum value for the number of docs we should return
            target          The target response we are trying to hit
            values          The list of values that we are sorting with
            n_sigmas        If we use a probability distribution function (e.g.,
                            Gaussian) to prioritize, then the PDF needs to have
                            a standard deviation associated with it. This standard
                            deviation is calculated by dividing the range in values
                            by `n_sigmas`.
        Output:
            parameters_list The list of parameters dictionaries that may be sent
                            to GASpy
        '''
        def __trim(docs, max_predictions):
            '''
            Trim the docs down according to this method's `max_predictions` argument. Since
            we trim the end of the list, we are implicitly prioritizing the docs in the
            beginning of the list.
            '''
            # Treat max_predictions==0 as no limit
            if max_predictions == 0:
                pass
            # TODO:  Address this if we ever address the top/bottom issue
            # We trim to half of max_predictions right now, because _make_parameters_list
            # currently creates two sets of parameters per system (i.e., top and bottom).
            # It's set up like this right now because our Local enumerated site DB is
            # not good at keeping track of top and bottom, so we do both (for now).
            else:
                docs = docs[:int(max_predictions/2)]
            return docs

        if len(docs) <= max_predictions/2:
            '''
            If we have less choices than the max number of predictions, then
            just move on
            '''
            pass

        elif prioritization == 'targeted':
            '''
            A 'targeted' prioritization means that we are favoring systems that predict
            values closer to our `target`.
            '''
            # And if the user chooses `targeted`, then they had better supply values
            if not values:
                raise Exception('Called the "targeted" prioritization without specifying values')
            # If the target was not specified, then just put it in the center of the range.
            if not target:
                target = (max(values)-min(values))/2.
            # `sort_inds` is a descending list of indices that correspond to the indices of
            # `values` that are proximate to `target`. In other words, `values[sort_inds[0]]`
            # is the closest value to `target`, and `values[sort_inds[-1]]` is furthest
            # from `target`. We use it to sort/prioritize the docs.
            sort_inds = sorted(range(len(values)), key=lambda i: abs(values[i]-target))
            docs = [docs[i] for i in sort_inds]
            docs = __trim(docs, max_predictions)

        elif prioritization == 'random':
            '''
            A 'random' prioritization means that we're just going to pick things at random.
            '''
            random.shuffle(docs)
            docs = __trim(docs, max_predictions)

        elif prioritization == 'gaussian':
            '''
            Here, we create a gaussian probability distribution centered at `target`. Then
            we choose points according to the probability distribution so that we get a lot
            of things near the target and fewer things the further we go from the target.
            '''
            # And if the user chooses `gaussian`, then they had better supply values.
            if not values:
                raise Exception('Called the "gaussian" prioritization without specifying values')
            # If the target was not specified, then just put it in the center of the range.
            if not target:
                target = (max(values)-min(values))/2.
            # `dist` is the distribution we use to choose our samples, and `pdf_eval` is a
            # list of probability density values for each of the energies. Google "probability
            # density functions" if you don't know how this works.
            dist = sp.stats.norm(target, (max(values)-min(values))/n_sigmas)
            pdf_eval = map(dist.pdf, values)
            # We use np.random.choice to do the choosing. But this function needs `p`, which
            # needs to sum to one. So we re-scale pdf_eval such that its sum equals 1; rename
            # it p, and call np.random.choice
            p = (pdf_eval/sum(pdf_eval)).tolist()
            docs = np.random.choice(docs, size=max_predictions, replace=False, p=p)

        else:
            raise Exception('User did not provide a valid prioritization')

        # Now create the parameters list from the trimmed and processed `docs`
        parameters_list = []
        for doc in docs:
            # Define the adsorption parameters via `defaults`. Then we change `numtosubmit`
            # to `None`, which indicates that we want to submit all of them.
            adsorption_parameters = defaults.adsorption_parameters(adsorbate=self.adsorbate,
                                                                   settings=self.calc_settings)
            adsorption_parameters['numtosubmit'] = None

            # Change the fingerprint to match the coordination of the doc we are looking at.
            # Since there is a chance the user may have omitted any of these fingerprints,
            # we use EAFP to define them.
            fp = {}
            try:
                fp['coordination'] = doc['coordination']
            except KeyError:
                pass
            try:
                fp['neighborcoord'] = doc['neighborcoord']
            except KeyError:
                pass
            try:
                fp['nextnearestcoordination'] = doc['nextnearestcoordination']
            except KeyError:
                pass
            adsorption_parameters['adsorbates'][0]['fp'] = fp

            # Add the parameters dictionary to our list for both the top and the bottom
            for top in [True, False]:
                slab_parameters = defaults.slab_parameters(miller=doc['miller'],
                                                           top=top,
                                                           shift=doc['shift'],
                                                           settings=self.calc_settings)
                # Finally:  Create the new parameters
                parameters_list.append({'bulk': defaults.bulk_parameters(doc['mpid'],
                                                                         settings=self.calc_settings),
                                        'gas': defaults.gas_parameters(self.adsorbate,
                                                                       settings=self.calc_settings),
                                        'slab': slab_parameters,
                                        'adsorption': adsorption_parameters})
        return parameters_list


    def anything(self, max_predictions=10):
        '''
        Call this method if you want n=`max_predictions` completely random things use in
        the next relaxation.

        Input:
            max_predictions     The number of random things you want to run.
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.site_docs` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the docs object to trim and use.
        docs = copy.deepcopy(self.site_docs)

        # Post-process the docs and make the parameters list
        parameters_list = self._make_parameters_list(docs,
                                                     prioritization='random',
                                                     max_predictions=max_predictions)

        return parameters_list


    def matching_ads(self, adsorbate, max_predictions=10):
        '''
        Call this method if you want n=`max_predictions` random sites that have already been
        relaxed with `adsorbate` on top. This method is useful for comparing a new adsorbate
        to an old one.

        Input:
            adsorbate           The adsorbate that you want to compare to.
            max_predictions     The number of random things you want to run.
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # Instead of looking at `self.site_docs`, we'll instead start with `self.ads_docs`,
        # since that list of docs contains relaxed docs
        docs = copy.deepcopy(self.ads_docs)
        # Filter out anything that doesn't include the adsorbate we're looking at.
        docs = [doc for doc in docs if doc['adsorbates'] == [adsorbate]]

        # Post-process the docs and make the parameters list
        parameters_list = self._make_parameters_list(docs,
                                                     prioritization='random',
                                                     max_predictions=max_predictions)

        return parameters_list


    def parameters(self, energy_min, energy_max, energy_target,
                   prioritization='gaussian', max_predictions=20):
        '''
        Input:
            energy_min      The lower-bound of the adsorption energy window that we want
                            to predict around (eV)
            energy_max      The upper-bound of the adsorption energy window that we want to
                            predict around (eV)
            energy_target   The adsorption energy we want to "hit" (eV)
            prioritization  A string that we pass to the `_sort` method. Reference that
                            method for more details.
            max_predictions A maximum value for the number of sets of `parameters` we should
                            return in `parameters_list`. If set to 0, then there is no limit.
        Output:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.site_docs` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the docs object to trim and use. Then we do the same for the energies
        docs = copy.deepcopy(self.site_docs)
        energies = copy.deepcopy(self.energies)

        # Trim the energies and mongo documents according to our energy boundaries
        energy_mask = (-(energy_min < np.array(energies))-(np.array(energies) < energy_max))
        docs = [docs[i] for i in np.where(energy_mask)[0].tolist()]
        energies = [energies[i] for i in np.where(energy_mask)[0].tolist()]

        # Post-process the docs; just read the method docstring for more details
        parameters_list = self._make_parameters_list(docs,
                                                     prioritization=prioritization,
                                                     max_predictions=max_predictions,
                                                     target=energy_target,
                                                     values=energies)

        return parameters_list
