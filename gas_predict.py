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
    # pylint: disable=too-many-instance-attributes
    def __init__(self, adsorbate,
                 pkl=None, features=None, block='no_block',
                 calc_settings='rpbe', fingerprints=None):
        '''
        Here, we open the pickle. And then depending on the model type, we assign different
        methods to use for predicting data. For reference:  These models are created by
        `regress.ipynb` in the `GASpy_regressions` submodule.

        We also initialize the `self.cat_docs` object, which will be a list of dictionaries
        containing calculation information for cataloged sites that have not yet been
        relaxed/put into the adsorption database.

        All of our __init__ arguments (excluding the pkl) are used to both define how we
        want to create our `parameters_list` objects and how we filter the `self.cat_docs` so
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
                            in this list. Refer to the `_preprocess_feature` method for a
                            list of the available feature strings.
            block           The block of the model that we should be using. See
                            GASpy_regression's `regress.ipynb` for more details regarding
                            what a `block` could be. It'll probably be a tuple.
            calc_settings   The calculation settings that we want to use. If we are using
                            something other than beef-vdw or rpbe, then we need to do some
                            more hard-coding here so that we know what in the local
                            database can work as a flag for this new calculation method.
            fingerprints    A dictionary of fingerprints and their locations in our
                            mongo documents. This is how we can pull out more (or less)
                            information from our database.
        Resulting attributes:
            adsorbate       A string representing the adsorbate that we want to make parameters for
            calc_settings   A string representing the calculation ssettings to use
                            (e.g., 'rpbe', 'beef-vdw')
            features        The same exact thing as the `features` argument
            cat_docs        A list of dictionaries (mongo documents) for each of the
                            elements in the catalog that is not yet in the "adsorption"
                            collection
            cat_docs        A list of dictionaries (mongo documents) for each of the
                            elements in the "adsorption" collection
            pre_processors  A dictionary containing pre-processing functions for the fingerprints
            norm            A np.array (vector) that contains the normalizing factors for
                            the pre-processed, stacked features
            model           A function created by the `__standardized_model` method that
                            accepts standardized inputs and returns a list of predictions
        '''
        # Default value for fingerprints. Since it's a mutable dictionary, we define it
        # down here instead of in the __init__ line.
        if not fingerprints:
            fingerprints = defaults.fingerprints()

        # Save some arguments to the object for later use
        self.adsorbate = adsorbate
        self.calc_settings = calc_settings
        self.features = features

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
        ads_hash = self.__hash_docs(ads_docs)
        cat_hash = list(self.__hash_docs(cat_docs).keys())
        # Perform the filtering while simultaneously creating `cat_docs`
        self.cat_docs = [doc for i, doc in enumerate(cat_docs)
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
            # Unpack and standardize the information in the pickled model
            with open(pkl, 'rb') as f:
                pkl = pickle.load(f)
            try:
                model = pkl['model'][block]
            except KeyError:
                raise Exception('The block %s is not in the model %s' %(block, pkl))
            self.pre_processors = pkl['pp']
            self.norm = pkl['norm']
            self.model = self.__standardize_model(model)
        else:
            print('No model provided. Only the "anything" or "matching_ads" methods will work.')


    def __hash_docs(self, docs):
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


    def __standardize_model(self, model):
        '''
        Each model type requires different methods/calls in order to be able to use
        them to make predictions. To make model usage easier, this method will return
        a function with a standardized input and standardized output. Below are the
        standardized inputs and outputs.

        Note that some models' inputs may deviate from this standard input. Refer
        to the specific functions' docstrings for details.

        Input:
            model       The model object that was saved in the pickle. Can vary in type.
        Output:
            standardized_model  A function that accepts a standard set of inputs and
                                outputs predictions.
        '''
        def sk_predict(inputs):
            ''' Assuming that the model in the pickle was an SKLearn model '''
            # SK models come with a method, `predict`, to transform the pre-processed input
            # to the output. Let's use it.
            return model.predict(inputs)

        # TODO:  Test this method
        def ala_predict(inputs):
            ''' Assuming that the model in the pickle was an alamopy model '''
            # Alamopy models come with a key, `f(model)`, that yields a lambda function to
            # transform the pre-processed input to the output. Let's use it.
            return model['f(model)'](inputs)

        # TODO:  Test this method
        def hierarch_predict(inputs_outer, inputs_inner):
            '''
            Assuming that the model in the pickle is a function created by GASpy_regression's
            `RegressionProcessor.hierarchical` method.

            Inputs:
                inputs_outer    A list of pre-processed factors/inputs that the outer model can accept.
                inputs_inner    A list of pre-processed factors/inputs that the outer model can accept.
            '''
            return model(inputs_outer, inputs_inner)

        # Figure out the model type and assign the correct function to perform predictions
        if isinstance(model, type(LinearRegression())):
            standardized_model = sk_predict
        elif isinstance(model, type(GradientBoostingRegressor())):
            standardized_model = sk_predict
        elif isinstance(model, type(GaussianProcessRegressor())):
            standardized_model = sk_predict
        elif isinstance(model, Pipeline):
            standardized_model = sk_predict
        elif isinstance(model, dict):
            standardized_model = ala_predict
        elif callable(model):
            standardized_model = hierarch_predict
        else:
            raise Exception('We have not yet established how to deal with this type of model')

        return standardized_model


    def _preprocess_feature(self, feature, docs):
        '''
        This method will pre-process whatever data the user wants. This helps turn
        "raw" data into numerical data that is accepted by models.

        Input:
            feature A string that represents the type of feature that
                    you want to pull out of the fingerprints. See the
                    `GASpy_regressions` repository for more details.
            docs    The mongo docs that you want to pre-process into features.
                    This should probably be either `self.cat_docs` or `self.ads_docs`.
        Output:
            p_data  A numpy array of pre-processed data. Each row represents
                    a new site, and each column represents a new numerical value
                    (depending on whatever the feature is).
        '''

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


    def _make_predictions(self, docs):
        '''
        Make predictions given a set of mongo documents. This is a method because
        we can make predictions on cataloged sites, already-done-relaxations, or both.

        Input:
            docs    Mongo documents of the sites you want to predict
        Output:
            predictions A list of predictions that `self.model` makes about `docs`
        '''
        # Pre-process, stack, and normalize the fingerprints into features so that
        # they may be passed as arguments to the model
        p_data = []
        for feature in self.features:
            p_data.append(self._preprocess_feature(feature, docs))
        model_inputs = np.array([np.hstack(_input) for _input in zip(*p_data)])/self.norm

        # Use the model to make predictions
        predictions = self.model(model_inputs)

        return predictions


    def _trim(self, docs, max_predictions):
        '''
        Trim the docs down according to this method's `max_predictions` argument. Since
        we trim the end of the list, we are implicitly prioritizing the docs in the
        beginning of the list.
        '''
        # Treat max_predictions == 0 as no limit
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


    def anything(self, max_predictions=20):
        '''
        Call this method if you want n=`max_predictions` completely random things use in
        the next relaxation.

        Input:
            max_predictions     The number of random things you want to run.
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.cat_docs` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the docs object to trim and use.
        docs = copy.deepcopy(self.cat_docs)

        # Post-process the docs and make the parameters list
        parameters_list = self._make_parameters_list(docs,
                                                     prioritization='random',
                                                     max_predictions=max_predictions)

        return parameters_list


    def matching_ads(self, adsorbate, max_predictions=20):
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
        # Instead of looking at `self.cat_docs`, we'll instead start with `self.ads_docs`,
        # since that list of docs contains relaxed docs
        docs = copy.deepcopy(self.ads_docs)
        # Filter out anything that doesn't include the adsorbate we're looking at.
        docs = [doc for doc in docs if doc['adsorbates'] == [adsorbate]]

        # Post-process the docs and make the parameters list
        parameters_list = self._make_parameters_list(docs,
                                                     prioritization='random',
                                                     max_predictions=max_predictions)

        return parameters_list


    def parameters(self, prediction_min, prediction_max, prediction_target,
                   prioritization='gaussian', max_predictions=20):
        # pylint: disable=too-many-branches, too-many-statements
        '''
        Input:
            prediction_min      The lower-bound of the adsorption prediction window that we want
                                to predict around (eV)
            prediction_max      The upper-bound of the adsorption prediction window that we want to
                                predict around (eV)
            prediction_target   The adsorption prediction we want to "hit" (eV)
            prioritization      A string that we pass to the `_sort` method. Reference that
                                method for more details.
            max_predictions     A maximum value for the number of sets of `parameters` we should
                                return in `parameters_list`. If set to 0, then there is no limit.
        Output:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.cat_docs` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the docs object to trim and use. Then we do the same for the predictions
        docs = copy.deepcopy(self.cat_docs)
        predictions = self._make_predictions(docs)

        # Trim the predictions and mongo documents according to our prediction boundaries
        prediction_mask = (-(prediction_min < np.array(predictions)) - \
                            (np.array(predictions) < prediction_max))
        docs = [docs[i] for i in np.where(prediction_mask)[0].tolist()]
        predictions = [predictions[i] for i in np.where(prediction_mask)[0].tolist()]


        if len(docs) <= max_predictions/2:
            ''' If we have less choices than the max number of predictions, then just move on '''

        elif prioritization == 'targeted':
            '''
            A 'targeted' prioritization means that we are favoring systems that predict
            values closer to our `target`.
            '''
            # And if the user chooses `targeted`, then they had better supply values
            if not values:
                raise Exception('Called the "targeted" prioritization without specifying values')
            # If the target was not specified, then just put it in the center of the range.
            if not prediction_target:
                prediction_target = (max(values)-min(values))/2.
            # `sort_inds` is a descending list of indices that correspond to the indices of
            # `values` that are proximate to `target`. In other words, `values[sort_inds[0]]`
            # is the closest value to `prediction_target`, and `values[sort_inds[-1]]` is furthest
            # from `prediction_target`. We use it to sort/prioritize the docs.
            sort_inds = sorted(range(len(values)), key=lambda i: abs(values[i]-prediction_target))
            docs = [docs[i] for i in sort_inds]
            docs = self._trim(docs, max_predictions)

        elif prioritization == 'random':
            '''
            A 'random' prioritization means that we're just going to pick things at random.
            '''
            random.shuffle(docs)
            docs = self._trim(docs, max_predictions)

        elif prioritization == 'gaussian':
            '''
            Here, we create a gaussian probability distribution centered at `prediction_target`.
            Then we choose points according to the probability distribution so that we get a lot
            of things near the prediction target and fewer things the further we go from the target.
            '''
            # And if the user chooses `gaussian`, then they had better supply values.
            if not values:
                raise Exception('Called the "gaussian" prioritization without specifying values')
            # If the target was not specified, then just put it in the center of the range.
            if not prediction_target:
                prediction_target = (max(values)-min(values))/2.
            # `dist` is the distribution we use to choose our samples, and `pdf_eval` is a
            # list of probability density values for each of the predictions. Google "probability
            # density functions" if you don't know how this works.
            dist = sp.stats.norm(prediction_target, (max(values)-min(values))/n_sigmas)
            pdf_eval = map(dist.pdf, values)
            # We use np.random.choice to do the choosing. But this function needs `p`, which
            # needs to sum to one. So we re-scale pdf_eval such that its sum equals 1; rename
            # it p, and call np.random.choice to do the selection. The `if-else` is to allow
            # us to pick everything if `max_predictions` == 0.
            p = (pdf_eval/sum(pdf_eval)).tolist()
            if max_predictions:
                docs = np.random.choice(docs, size=max_predictions, replace=False, p=p)
            else:
                docs = np.random.choice(docs, size=len(docs), replace=False, p=p)

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


    def values(self, collection='catalog'):
        '''
        This method will predict adsorption predictions of un-relaxed sites in the site catalog.

        Input:
            collection  A string for the mongo db collection that you want to get predictions for.
                        Should probably be either 'catalog' or 'adsorption'.
        Output:
            data    A list of 2-tuples for each un-simulated site. The first tuple
                    element is the mongo doc of the site (as per the `fingerprints`
                    argument in `__init__`), and the second tuple element is
                    the model's prediction.
        '''
        # Make and collect predictions for the catalog collection
        if collection == 'catalog':
            docs = self.cat_docs
            predictions = self._make_predictions(docs)

        # Make and collect predictions for the adsorption collection
        if collection == 'adsorption':
            docs = self.ads_docs
            predictions = self._make_predictions(docs)

        data = zip(self.cat_docs, predictions)
        return data
