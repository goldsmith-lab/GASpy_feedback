'''
When given the location for a pickled GASpy_regressions model, this class will open the pickle
and create a list of GASpy `parameters` to run.
'''
__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'
import sys
import copy
from random import shuffle
from pprint import pprint   # for debugging
import numpy as np
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
#DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB'    # Cori
DB_LOC = '/Users/Ktran/Nerd/GASpy'                  # Local

class GASPredict(object):
    def __init__(self, pkl):
        '''
        First, open the pickle. And then depending on the model type, we assign different
        methods to use for predicting data.
        For reference:  These models are created by `regress.ipynb` in the `GASpy_regressions`
        submodule.

        Input:
            pkl  A pickled dictionary object
                'model' key should contain the model object.
                'pre_processors' key should contain a dictionary whose keys are the
                    model features (i.e., the ase-db row attributes) and whose values
                    are the already-fitted pre-processors (e.g., LabelBinarizer)
        '''
        # For now, we will only "predict" (i.e., suggest) relaxations of systems that have
        # been enumerated and stored in `enumerated adsorption sites.db`. As such, we first
        # pull out information from that database. We also filter out things we have already
        # run, which should be stored in the `odsorption_energy_database.db`.
        sites_db = connect(DB_LOC+'/enumerated_adsorption_sites.db')
        enrgs_db = connect(DB_LOC+'/adsorption_energy_database.db')
        sites_rows = [row for row in sites_db.select()]
        enrgs_rows = [row for row in enrgs_db.select()]

        # The `rows` object will be the set of adsorption sites that we will be considering
        # for feedback. Here, we initialize it as the full set of adsorption sites, and then
        # we remove any sites that occur in adsorption energy database.
        self.rows = sites_rows
        for site_row in sites_rows:
            for enrg_row in enrgs_rows:
                if (site_row.mpid == enrg_row.mpid
                        and site_row.miller == enrg_row.miller
                        and site_row.adsorption_site == enrg_row.adsorption_site):
                    self.rows.remove(site_row)
                    break

        # Unpack the pickle
        pkl = pickle.load(open(pkl, 'r'))
        self.model = pkl['model']
        self.pre_processors = pkl['pre_processors']

        # Figure out the model type and assign the correct method to perform predictions
        if isinstance(self.model, type(LinearRegression())):
            self.predict = self._sk_predict
        elif isinstance(self.model, type(GradientBoostingRegressor())):
            self.predict = self._sk_predict
        else:
            raise Exception('We have not yet established how to deal with this type of model')


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


    def _sort(self, rows, prioritization, target='foo', values='bar'):
        '''
        This method will rearrange a list of ase-db row objects according to a
        user-specified method.

        Inputs:
            rows            A list of ase-db row objects
            prioritization  A string corresponding to a particular prioritization method.
                            So far, valid values include:
                                targeted (try to hit a single value, `target`)
                                random (randomly chosen)
                                gaussian (gaussian spread around target)
            target          The target response we are trying to hit
            values          The list of values that we are sorting with
        Output:
            rows    The rearrange version of the supplied list
        '''

        # A 'targeted' prioritization means that we are favoring systems
        # that predict values closer to our `target`.
        if prioritization == 'targeted':
            # If the user chooses `targeted`, then they had better pick their target
            if (target == 'foo' or values == 'bar'):
                raise Exception('Called the "centering" prioritization without specifying arguments')
            # `sort_inds` is a descending list of indices that correspond to the indices of
            # `values` that are proximate to `target`. In other words, `values[sort_inds[0]]`
            # is the closest value to `target`, and `values[sort_inds[-1]]` is furthest
            # from `target`.
            sort_inds = sorted(range(len(values)), key=lambda i: abs(values[i]-target))
            return [rows[i] for i in sort_inds]

        # A 'random' prioritization means that we're just going to pick things at random.
        elif prioritization == 'random':
            return shuffle(rows)

        # A 'gaussian' prioritization is a bit of a mix between 'centered' and 'random'. Here,
        # we create a gaussian probability distribution centered at `target`. Then
        # we choose points according to the probability distribution so that we get a lot
        # of things near the target and fewer thnigs the further we go from the target.
        elif prioritization == 'gaussian':
            # If the user chooses `gaussia`, then they had better pick their center/target
            if (target == 'foo' or values == 'bar'):
                raise Exception('Called the "centering" prioritization without specifying arguments')
            raise Exception('Yeah... still figuring out Gaussian prioritization')
            return rows

        else:
            raise Exception('User did not provide a valid prioritization')


    def coordcount_ads_to_energy(self, adsorbate='CO',
                                 prioritization='targeted', n_predictions=0,
                                 energy_min=-0.7, energy_max=-0.5):
        '''
        The user must call this method to return the list of GASpy `parameters` objects
        to run.

        Input:
            adsorbate       A string for the adsorbate to be predicting. If you want more than
                            one adsorbate, then loop over this method.
            prioritization  A string that we pass to the `_sort` method. Reference that
                            method for more details.
            n_predictions   A maximum value for the number of sets of `parameters` we should
                            return in `parameters_list`. If set to 0, then there is no limit.
            energy_min      The lower-bound of the adsorption energy window that we want
                            to predict around
            energy_max      The upper-bound of the adsorption energy window that we want to
                            predict around
        Outut:
            parameters_list     A list of `parameters` dictionaries that we may pass
                                to GASpy
        '''
        # We will be trimming the `self.rows` object. But in case the user wants to use
        # the same class instance to call a different method, we create a local copy of
        # the rows object to trim and use.
        rows = copy.deepcopy(self.rows)

        # Pre-process the coordination and the adsorbate, and then stack them together so that
        # they may be accepted as direct inputs to the model.
        p_coord = np.array([np.sum(self.pre_processors['coordination'].transform(coord.split('-')), axis=0)
                            for coord in [row.coord for row in rows]])
        p_ads = np.array(adsorbate)
        p_input = np.hstack(p_coord, p_ads)
        # Predict the adsorption energies of our rows, and then trim any that lie outside
        # the specified range. Note that we trim both the rows and their corresponding energies
        energies = self.predict(p_input)
        energy_mask = (-(energy_min < np.array(energies))-(np.array(energies) < energy_max)).tolist()
        rows = rows[energy_mask]
        energies = energies[energy_mask]

        # Rearrange the rows so that the higher priority rows are earlier in the list
        rows = self._sort(rows, prioritization=prioritization,
                          target=(energy_min+energy_max)/2,
                          values=energies)
        # Trim the rows down according to this method's `n_predictions` arugment. Since
        # we trim the end of the list, we are implicitly prioritizing the rows in the
        # beginnig of the list (thus the reason for _sort).
        if n_predictions == 0:  # We treat 0 as no limit
            pass
        else:
            rows = rows[:n_predictions]

        # Create a parameters_list from our rows list.
        parameters_list = []
        for row in rows:
            parameters_list.append({'bulk': dpar_bulk(row.mpid),
                                    'slab': dpar_slab(miller=row.miller,
                                                      top=row.top,
                                                      shift=row.shift),
                                    'gas': dpar_gas(adsorbate),
                                    'adsorption': dpar_ads(adsorbate=adsorbate,
                                                           adsorption_site=row.adsorption_site)})

        # We're done!
        return parameters_list
