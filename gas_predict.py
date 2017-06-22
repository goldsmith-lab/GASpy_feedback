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
import numpy as np
import scipy as sp
import dill as pickle
pickle.settings['recurse'] = True     # required to pickle lambdify functions
from ase.db import connect
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
sys.path.append('..')
from gaspy_toolbox import default_parameter_bulk as dpar_bulk
from gaspy_toolbox import default_parameter_slab as dpar_slab
from gaspy_toolbox import default_parameter_gas as dpar_gas
from gaspy_toolbox import default_parameter_adsorption as dpar_ads
sys.path.append('../GASpy_regressions')


# The location of the Local database. Do not include the name of the file.
DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB'    # Cori
#DB_LOC = '/Users/Ktran/Nerd/GASpy'                  # Local


class GASPredict(object):
    def __init__(self, pkl, adsorbate, calc_settings='beef-vdw'):
        '''
        Here, we open the pickle. And then depending on the model type, we assign different
        methods to use for predicting data. For reference:  These models are created by
        `regress.ipynb` in the `GASpy_regressions` submodule.

        We also initialize the `self.rows` object, which will be a list of ase-db rows
        (from our Local adsorption site DB) whose corresponding sites we predict will be
        good ones to relax next.

        All of our __init__ arguments (excluding the pkl) are used to both define how we
        want to create our `parameters_list` objects and how we filter the `self.rows` so
        that we don't try to re-submit calculations that we've already done.

        Input:
            pkl             The location of a pickle dictionary object
                            'model' key should contain the model object.
                            'pre_processors' key should contain a dictionary whose keys are the
                                model features (i.e., the ase-db row attributes) and whose values
                                are the already-fitted pre-processors (e.g., LabelBinarizer)
                            Note that the user may provide "None" or "False" for this argument,
                            as well. This should be done if the user wants to call the "anything"
                            method.
            adsorbate       A string indicating the adsorbate that you want to make a prediction
                            for.
            calc_settings   The calculation settings that we want to use. If we are using
                            something other than beef-vdw or rpbe, then we need to do some
                            more hard-coding here so that we know what in the local energy
                            database can work as a flag for this new calculation method.
        '''
        # Save some arguments to the object for later use
        self.adsorbate = adsorbate
        self.calc_settings = calc_settings

        # We import the adsorption site DB to initialize `rows` (see __init__'s docstring).
        # We also import the adsorption energy DB so that we can filter out rows that we've
        # already relaxed.
        with connect(DB_LOC+'/adsorption_energy_database.db') as enrgs_db:
            with connect(DB_LOC+'/enumerated_adsorption_sites.db') as sites_db:
                sites_rows = [row for row in sites_db.select()]
                # Filter out beef-vdw vs rpbe
                if calc_settings == 'beef-vdw':
                    enrgs_rows = [row for row in enrgs_db.select() if row.gga == 'BF']
                elif calc_settings == 'rpbe':
                    enrgs_rows = [row for row in enrgs_db.select() if row.gga == 'RP']
                else:
                    raise Exception('Unknown calculations calculation settings')
                '''
                Both databases are large and take a long time to filter. The best way to
                filter them (that we can think of) is to hash the important features of
                each system (see the _hash_row method), and then store these hashed
                system into dictionary keys, which can be searched/parsed quickly.
                We can then check hashes against each other in this dictionary.
                '''
                # Create `relaxed_systems_dict`, whose keys are the hashes of the systems
                # we have relaxed and stored in the local energy DB
                relaxed_systems = np.unique(map(self._hash_row,
                                                [(row, adsorbate) for row in enrgs_rows]))
                relaxed_systems_dict = {}
                for relaxed_system in relaxed_systems:
                    relaxed_systems_dict[relaxed_system] = None
                # Now initialize/filter `self.rows`
                self.rows = [site_row for site_row in sites_rows
                             if self._hash_row((site_row, adsorbate)) not in relaxed_systems_dict]

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
            elif isinstance(self.model, dict):
                self.predict = self._ala_predict
            else:
                raise Exception('We have not yet established how to deal with this type of model')
        else:
            print('No model provided. Only the "anything" method will work')


    def _hash_row(self, tup):
        '''
        This method helps convert the important characteristics of our systems into hashes
        so that we may sort through them more quickly. This is important to do when trying to
        compare entries in our two databases; it helps speed things up. Check out the __init__
        method for more details.

        Input:
            tup     A 2-tuple. The first item should be the row object you are hashing, and
                    the second item should be the adsorbate you are interested in. Note that
                    this method effectively ignores the second "adsorbate" item if it thinks
                    that you've passed it a row object from the Local energy database.
        '''
        # Unpack the tuple
        row = tup[0]
        adsorbate = tup[1]
        # Unpack information from the row
        mpid = row.mpid
        shift = row.shift

        # Use EAFP (Google it) to unpack informaion from either the energy database or the site
        # database (respectively), since they have slightly different attribute characteristics
        try:
            adsorbate = row.adsorbate
            miller = ' '.join(row.miller.split('.'))
            coordination = row.initial_coordination
            neighborcoord = row.initial_neighborcoord
            nextnearestcoordination = row.initial_nextnearestcoordination
        except AttributeError:
            adsorbate = self.adsorbate
            miller = ' '.join(row.miller.split(', '))
            coordination = row.coordination
            neighborcoord = row.neighborcoord
            nextnearestcoordination = row.nextnearestcoordination

        # Put the surface characteristics together into one string, `system` so that we can hash it
        system = ''
        for item in [adsorbate, mpid, miller, shift,
                     coordination, neighborcoord, nextnearestcoordination]:
            system += str(item)
        return hash(system)


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
        def _trim(rows, max_predictions):
            '''
            Trim the rows down according to this method's `max_predictions` argument. Since
            we trim the end of the list, we are implicitly prioritizing the rows in the
            beginning of the list.
            '''
            # Treat max_predictions==0 as no limit
            if max_predictions == 0:
                pass
            else:
                rows = rows[:max_predictions]
            return rows

        # If we have less choices than the max number of predictions, then just give it back...
        if len(rows) <= max_predictions:
            return rows

        elif prioritization == 'targeted':
            # A 'targeted' prioritization means that we are favoring systems that predict
            # values closer to our `target`. And if the user chooses `targeted`, then they
            # had better supply values
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
            return _trim(rows, max_predictions)

        elif prioritization == 'random':
            # A 'random' prioritization means that we're just going to pick things at random.
            random.shuffle(rows)
            return _trim(rows, max_predictions)

        elif prioritization == 'gaussian':
            # Here, we create a gaussian probability distribution centered at `target`. Then
            # we choose points according to the probability distribution so that we get a lot
            # of things near the target and fewer thnigs the further we go from the target.
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
            # needs to sum to one. So we re-scale pdf_eval such that its sum equals 1; then
            # we call it p.
            p = (pdf_eval/sum(pdf_eval)).tolist()
            return np.random.choice(rows, size=max_predictions, replace=False, p=p)

        else:
            raise Exception('User did not provide a valid prioritization')


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
        Call this method if you want `max_predictions` completely random things to run.
        Input:
            max_predictions     The number of random things you want to run.
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.rows` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the rows object to trim and use.
        rows = copy.deepcopy(self.rows)

        # Post-process the rows; just read the method docstring for more details
        rows = self._post_process(rows,
                                  prioritization='random'
                                  max_predictions=max_predictions)

        # Create a parameters_list from our rows list.
        parameters_list = []
        for row in rows:
            parameters_list.append({'bulk': dpar_bulk(row.mpid,
                                                      settings=self.calc_settings),
                                    'gas': dpar_gas(self.adsorbate,
                                                    settings=self.calc_settings),
                                    'slab': dpar_slab(miller=row.miller,
                                                      top=row.top,
                                                      shift=row.shift,
                                                      settings=self.calc_settings),
                                    'adsorption': dpar_ads(adsorbate=self.adsorbate,
                                                           adsorption_site=row.adsorption_site,
                                                           settings=self.calc_settings)})
        # We're done!
        return parameters_list


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
        # We will be trimming the `self.rows` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the rows object to trim and use.
        rows = copy.deepcopy(self.rows)
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

        # Create a parameters_list from our rows list.
        parameters_list = []
        for row in rows:
            parameters_list.append({'bulk': dpar_bulk(row.mpid,
                                                      settings=self.calc_settings),
                                    'gas': dpar_gas(self.adsorbate,
                                                    settings=self.calc_settings),
                                    'slab': dpar_slab(miller=row.miller,
                                                      top=row.top,
                                                      shift=row.shift,
                                                      settings=self.calc_settings),
                                    'adsorption': dpar_ads(adsorbate=self.adsorbate,
                                                           adsorption_site=row.adsorption_site,
                                                           settings=self.calc_settings)})
        # We're done!
        return parameters_list
