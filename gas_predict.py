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
sys.path.insert(0, '../')
from gaspy import defaults
from gaspy import utils
sys.path.insert(0, '../GASpy_regressions')


# The location of the Local database. Do not include the name of the file.
DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB'    # Cori
#DB_LOC = '/Users/KTran/Nerd/GASpy'                  # Local

class GASPredict(object):
    def __init__(self, adsorbate, pkl=None, calc_settings='beef-vdw',
                 fingerprints={'mpid': '$processed_data.calculation_info.mpid',
                               'miller': '$processed_data.calculation_info.miller',
                               'shift': '$processed_data.calculation_info.shift',
                               'top': '$processed_data.calculation_info.top',
                               'coordination': '$processed_data.fp_init.coordination',
                               'neighborcoord': '$processed_data.fp_init.neighborcoord',
                               'nextnearestcoordination': '$processed_data.fp_init.nextnearestcoordination'}):
        '''
        Here, we open the pickle. And then depending on the model type, we assign different
        methods to use for predicting data. For reference:  These models are created by
        `regress.ipynb` in the `GASpy_regressions` submodule.

        We also initialize the `self.site_docs` object, which will be a list of dictionaries
        containing calculation information for cataloged sites that have not yet been
        relaxed/put into the adsorption database.

        All of our __init__ arguments (excluding the pkl) are used to both define how we
        want to create our `parameters_list` objects and how we filter the `self.site_rows` so
        that we don't try to re-submit calculations that we've already done.

        Input:
            pkl             The location of a pickle dictionary object:
                                'model' key should contain the model object.
                                'pre_processors' key should contain a dictionary whose keys
                                 are the model features (i.e., the ase-db row attributes) and
                                 whose values are the already-fitted pre-processors
                                 (e.g., LabelBinarizer)
                            Note that the user may provide '' for this argument, as well. This
                            should be done if the user wants to call the "anything" method.
            adsorbate       A strings indicating the adsorbate that you want to make a
                            prediction for.
            calc_settings   The calculation settings that we want to use. If we are using
                            something other than beef-vdw or rpbe, then we need to do some
                            more hard-coding here so that we know what in the local energy
                            database can work as a flag for this new calculation method.
            fingerprints    A dictionary of fingerprints and their locations in our
                            mongo documents.
        '''
        # Save some arguments to the object for later use
        self.adsorbate = adsorbate
        self.calc_settings = calc_settings

        # Fetch mongo cursors for our adsorption and catalog databases so that we can
        # start parsing.
        with utils.get_adsorption_db() as ads_db:
            ads_cursor = self._get_cursor(ads_db, 'adsorption',
                                          calc_settings=calc_settings,
                                          fingerprints=fingerprints)
        with utils.get_catalog_db() as cat_db:
            cat_cursor = self._get_cursor(cat_db, 'catalog', fingerprints=fingerprints)
        # Create copies of the catalog cusror (generator) since we'll be using
        # it more than once.
        cat_cursor, _cat_cursor = itertools.tee(cat_cursor)
        # Hash the cursors so that we can filter out any items in the catalog
        # that we have already relaxed and put into the adsorption database.
        # Note that we keep `ads_hash` in a dict so that we can search through
        # it, but we turn `cat_hash` into a list so that we can iterate through
        # it alongside `cat_cursor`
        ads_hash = self._hash_cursor(ads_cursor)
        cat_hash = list(self._hash_cursor(cat_cursor).keys())
        # Perform the filtering while simultaneously creating a big
        # fingerprint thing...?
        self.site_docs = [doc['_id'] for i, doc in enumerate(_cat_cursor)
                          if cat_hash[i] not in ads_hash]

        # We put a conditional before we start working with the pickled model. This allows the
        # user to go straight for the "anything" method without needing to find a dummy model.
        if pkl:
            # Unpack the pickled model
            pkl = pickle.load(open(pkl, 'r'))
            self.model = pkl['model']
            self.pre_processors = pkl['pre_processors']

            # Figure out the model type and assign the correct method to perform predictions
            if isinstance(self.model, type(LinearRegression())):
                self.predict = self._sk_predict
            elif isinstance(self.model, type(GradientBoostingRegressor())):
                self.predict = self._sk_predict
            elif isinstance(self.model, type(GaussianProcessRegressor())):
                self.predict = self._sk_predict
            elif isinstance(self.model, dict):
                self.predict = self._ala_predict
            else:
                raise Exception('We have not yet established how to deal with this type of model')
        else:
            print('No model provided. Only the "anything" method will work.')
        pdb.set_trace()


    def _get_cursor(self, client, collection_name, fingerprints, calc_settings=None):
        '''
        This method pulls out a set of fingerprints from a mongo client and returns
        a mongo cursor (generator) object that returns the fingerprints

        Inputs:
            client              Mongo client object
            collection_name     The collection name within the client that you want to look at
            fingerprints        A dictionary of fingerprints and their locations in our
                                mongo documents. For example:
                                    fingerprints = {'mpid': '$processed_data.calculation_info.mpid',
                                                    'coordination': '$processed_data.fp_init.coordination'}
            calc_settings       An optional argument that will only pull out data with these
                                calc settings.

        Output:
            cursor  A mongo cursor object that can be iterated to return a dictionary
                    of fingerprint properties
        '''
        # Put the "fingerprinting" into a `group` dictionary, which we will
        # use to pull out data from the mongo database
        group = {'$group': {'_id': fingerprints}}

        # If the user did not put in calc_settings, then assume they have passed
        # a catalog collection and proceed as normal.
        if not calc_settings:
            pipeline = [group]

        # If the user provided calc_settings, then match only results that use
        # this calc_setting. We also assume that this collection is an
        # adsorption collection, in which case we also need to match results
        # that use the adsorbate(s) we are looking at.
        elif calc_settings == 'rpbe':
            match = {'$match': {'processed_data.vasp_settings.gga': 'RP',
                                'processed_data.calculation_settings.adsorbate_names': [self.adsorbate]}}
            pipeline = [match, group]
        elif calc_settings == 'beef-vdw':
            match = {'$match': {'processed_data.vasp_settings.gga': 'BF',
                                'processed_data.calculation_settings.adsorbate_names': [self.adsorbate]}}
            pipeline = [match, group]
        else:
            raise Exception('Unknown calc_settings')

        # Get the particular collection from the mongo client's database
        collection = getattr(client.db, collection_name)

        # Create the cursor. We set allowDiskUse=True to allow mongo to write to
        # temporary files, which it needs to do for large databases. We also
        # set useCursor=True so that `aggregate` returns a cursor object
        # (otherwise we run into memory issues).
        cursor = collection.aggregate(pipeline, allowDiskUse=True, useCursor=True)
        return cursor


    def _hash_cursor(self, cursor):
        '''
        This method helps convert the important characteristics of our systems into hashes
        so that we may sort through them more quickly. This is important to do when trying to
        compare entries in our two databases; it helps speed things up. Check out the __init__
        method for more details.

        Note that this method will consume whatever iterator it is given.

        Input:
            cursor      A pymongo cursor object that has been created using `_get_cursor`.
                        This method assumes that the `cursor` object was created by
                        `collection.aggregate()`, because cursors from `collection.aggregate()`
                        return dictionaries that are nested within the `_id` key.
                        Normally-generated cursors will return actual IDs, not dicts.
        Output:
            systems     An ordered dictionary whose keys are hashes of the mongo doc
                        returned by `cursor` and whose values are empty. This dictionary
                        is intended to be parsed alongside the cursor, which is why it's
                        ordered.
        '''
        systems = OrderedDict()
        for doc in cursor:
            # `system` will be one long string of the fingerprints
            system = ''
            for key, value in doc['_id'].iteritems():
                # Note that we turn the values into strings explicitly, because some
                # fingerprint features may not be strings (e.g., list of miller indices).
                system += str(key + '=' + str(value) + ', ')
            systems[hash(system)] = None

        return systems


    def _post_process(self, rows, prioritization, max_predictions,
                      target=None, values=None, n_sigmas=6):
        '''
        Given the remaining ase-db row objects, this method will decide which of those
        rows to return for further relaxations. We do this in two steps:  1) choose and
        use a prioritization method (i.e., how to pick the rows), and then 2) trim the
        rows down to the number of predictions we want.

        Inputs:
            rows            A list of ase-db row objects
            prioritization  A string corresponding to a particular prioritization method.
                            So far, valid values include:
                                targeted (try to hit a single value, `target`)
                                random (randomly chosen)
                                gaussian (gaussian spread around target)
            max_predictions A maximum value for the number of rows we should return
            target          The target response we are trying to hit
            values          The list of values that we are sorting with
            n_sigmas        If we use a probability distribution function (e.g.,
                            Gaussian) to prioritize, then the PDF needs to have
                            a standard deviation associated with it. This standard
                            deviation is calculated by dividing the range in values
                            by `n_sigmas`.
        Output:
            rows    The rearrange version of the supplied list
        '''
        def __trim(rows, max_predictions):
            '''
            Trim the rows down according to this method's `max_predictions` argument. Since
            we trim the end of the list, we are implicitly prioritizing the rows in the
            beginning of the list.
            '''
            # Treat max_predictions==0 as no limit
            if max_predictions == 0:
                pass
            # We trim to half of max_predictions right now, because _make_parameters_list
            # currently creates two sets of parameters per system (i.e., top and bottom).
            # It's set up like this right now because our Local enumerated site DB is
            # not good at keeping track of top and bottom, so we do both (for now).
            else:
                rows = rows[:int(max_predictions/2)]
            return rows

        if len(rows) <= max_predictions:
            '''
            If we have less choices than the max number of predictions, then
            just give it back...
            '''
            return rows

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
            # from `target`. We use it to sort/prioritize the rows.
            sort_inds = sorted(range(len(values)), key=lambda i: abs(values[i]-target))
            rows = [rows[i] for i in sort_inds]
            return __trim(rows, max_predictions)

        elif prioritization == 'random':
            '''
            A 'random' prioritization means that we're just going to pick things at random.
            '''
            random.shuffle(rows)
            return __trim(rows, max_predictions)

        elif prioritization == 'gaussian':
            '''
            Here, we create a gaussian probability distribution centered at `target`. Then
            we choose points according to the probability distribution so that we get a lot
            of things near the target and fewer thnigs the further we go from the target.
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
            return np.random.choice(rows, size=max_predictions, replace=False, p=p)

        else:
            raise Exception('User did not provide a valid prioritization')


    def _make_parameters_list(self, rows):
        parameters_list = []
        for row in rows:
            # Define the adsorption parameters via `defaults`. Then we change `numtosubmit`
            # to `None`, which indicates that we want to submit all of them. Also change
            # the fingerprint to match the coordination of the row we are looking at.
            adsorption_parameters = defaults.adsorption_parameters(adsorbate=self.adsorbate,
                                                                   settings=self.calc_settings)
            adsorption_parameters['numtosubmit'] = None
            adsorption_parameters['adsorbates'][0]['fp'] = {'coordination': row.coordination,
                                                            'nextnearestcoordination': row.nextnearestcoordination}
            # Add the parameters dictionary to our list for both the top and the bottom
            for top in [True, False]:
                # Define the slab parameters. We may be pulling the miller indices from
                # either the Local site DB or the Local energy DB, which have different
                # formats for miller index (i.e., (1.1.1) or (1, 1, 1)), we use EAFP to
                # process either of them.
                try:
                    slab_parameters = defaults.slab_parameters(miller=[int(ind)
                                                                       for ind in row.miller[1:-1].split(', ')],
                                                               top=top,
                                                               shift=row.shift,
                                                               settings=self.calc_settings)
                except ValueError:
                    slab_parameters = defaults.slab_parameters(miller=[int(ind)
                                                                       for ind in row.miller[1:-1].split('.')],
                                                               top=top,
                                                               shift=row.shift,
                                                               settings=self.calc_settings)
                # Finally:  Create the new parameters
                parameters_list.append({'bulk': defaults.bulk_parameters(row.mpid,
                                                                         settings=self.calc_settings),
                                        'gas': defaults.gas_parameters(self.adsorbate,
                                                                       settings=self.calc_settings),
                                        'slab': slab_parameters,
                                        'adsorption': adsorption_parameters})
        return parameters_list


    def _sk_predict(self, inputs):
        '''
        Assuming that the model in the pickle was an SKLearn model, predict the model's
        output given the pre-processed inputs

        Input:
            inputs      A list of pre-processed factors/inputs that the model can accept
        Output:
            responses   A list of values that the model predicts for the rows
        '''
        # SK models come with a method, `predict`, to transform the pre-processed input
        # to the output. Let's use it.
        return self.model.predict(inputs)


    # TODO:  Test this method
    def _ala_predict(self, inputs):
        '''
        Assuming that the model in the pickle was an alamopy model, predict the model's
        output given the pre-processed inputs

        Input:
            inputs      A list of pre-processed factors/inputs that the model can accept
        Output:
            responses   A list of values that the model predicts for the rows
        '''
        # Alamopy models come with a key, `f(model)`, that yields a lambda function to
        # transform the pre-processed input to the output. Let's use it.
        return self.model['f(model)'](inputs)


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
        # We will be trimming the `self.site_rows` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the rows object to trim and use.
        rows = copy.deepcopy(self.site_rows)

        # Post-process the rows; just read the method docstring for more details
        rows = self._post_process(rows,
                                  prioritization='random',
                                  max_predictions=max_predictions)

        # Use the _make_parameters_list method to turn the list of rows into a list of parameters
        return self._make_parameters_list(rows)


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
        # Instead of looking at `self.site_rows`, we'll instead start with `self.energy_rows`,
        # since that list of rows contains relaxed rows
        rows = copy.deepcopy(self.energy_rows)
        # Filter out anything that doesn't include the adsorbate we're looking at.
        rows = [row for row in rows if row.adsorbate == adsorbate]

        # Post-process the rows; just read the method docstring for more details
        rows = self._post_process(rows,
                                  prioritization='random',
                                  max_predictions=max_predictions)

        # Use the _make_parameters_list method to turn the list of rows into a list of parameters
        return self._make_parameters_list(rows)


    def energy_fr_coordcount_ads(self, prioritization='gaussian', max_predictions=0,
                                 energy_min=-0.7, energy_max=-0.5, energy_target=-0.6):
        '''
        The user must call this method to return the list of GASpy `parameters` objects
        to run.

        Input:
            prioritization  A string that we pass to the `_sort` method. Reference that
                            method for more details.
            max_predictions A maximum value for the number of sets of `parameters` we should
                            return in `parameters_list`. If set to 0, then there is no limit.
            energy_min      The lower-bound of the adsorption energy window that we want
                            to predict around (eV)
            energy_max      The upper-bound of the adsorption energy window that we want to
                            predict around (eV)
            energy_target   The adsorption energy we want to "hit" (eV)
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.site_rows` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the rows object to trim and use.
        rows = copy.deepcopy(self.site_rows)
        # Pull in the pre-processors. This in not a necessary step, but it might help
        # readability later on.
        lb_coord = self.pre_processors['coordination']
        lb_ads = self.pre_processors['adsorbate']

        # Pre-process the coordination and the adsorbate, and then stack them together so that
        # they may be accepted as direct inputs to the model. This should be identical to the
        # pre-processing performed in GASpy_regresson's `regress.ipynb`. But it isn't, because
        # numpy is stupid and we had to reshape/list things that are totally not intuitive.
        # Just go with it. Don't worry about it.
        p_coords = np.array([np.sum(lb_coord.transform(coord.split('-')), axis=0)
                             for coord in [row.coordination for row in rows]])
        p_ads = lb_ads.transform([self.adsorbate])[0]
        p_inputs = np.array([np.hstack((p_coord, p_ads)) for p_coord in p_coords])
        # Predict the adsorption energies of our rows, and then trim any that lie outside
        # the specified range. Note that we trim both the rows and their corresponding energies
        energies = self.predict(p_inputs)
        energy_mask = (-(energy_min < np.array(energies))-(np.array(energies) < energy_max))
        rows = [rows[i] for i in np.where(energy_mask)[0].tolist()]
        energies = [energies[i] for i in np.where(energy_mask)[0].tolist()]

        # Post-process the rows; just read the method docstring for more details
        rows = self._post_process(rows,
                                  prioritization=prioritization,
                                  max_predictions=max_predictions,
                                  target=energy_target,
                                  values=energies)

        # Use the _make_parameters_list method to turn the list of rows into a list of parameters
        return self._make_parameters_list(rows)
