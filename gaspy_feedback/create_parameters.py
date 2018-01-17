'''
This module is meant to be used to close GASpy's feedback loop
by establishing which simulations to run next.

Here are the common inputs/outputs for all of the non-hidden functions.

Input:
    adsorbate       A string indicating the adsorbate that you want to make a
                    prediction for.
    calc_settings   The calculation settings that we want to use. If we are using
                    something other than beef-vdw or rpbe, then we need to do some
                    more hard-coding here so that we know what in the catalog
                    can work as a flag for this new calculation method.
    max_predictions An integer representing the maximum number of
                    `parameter` dictionaries that you want back
    max_atoms       The maximum number of atoms in the system that you want to pull
Output:
    parameters_list A list of `parameters` dictionaries that can be
                    passed to GASpy to execute a simulation
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import pdb  # noqa: F401
import random
from collections import OrderedDict
import numpy as np
import scipy as sp
import dill as pickle
import re
from gaspy import defaults, gasdb, utils
from gaspy_regress.regressor import GASpyRegressor  # noqa: F401
pickle.settings['recurse'] = True     # required to pickle lambdify functions


def randomly(adsorbate, calc_settings='rpbe', max_predictions=20, max_atoms=50):
    ''' Call this method if you want n=`max_predictions` completely random things '''
    docs = gasdb.unsimulated_catalog(adsorbate,
                                     calc_settings=calc_settings,
                                     max_atoms=max_atoms)
    parameters_list = _make_parameters_list(docs, [adsorbate],
                                            prioritization='random',
                                            max_predictions=max_predictions,
                                            calc_settings=calc_settings,
                                            max_atoms=max_atoms)
    return parameters_list


def from_matching_ads(adsorbate, matching_ads, calc_settings='rpbe',
                      max_predictions=20, max_atoms=50):
    '''
    Call this method if you want n=`max_predictions` random sites that have already been
    relaxed with `adsorbate` on top. This method is useful for comparing a new adsorbate
    to an old one.

    Special input:
        matching_ads    The adsorbate that you want to compare to.
    '''
    # Find a list of the simulations that we haven't done yet, `cat_docs`
    cat_docs = gasdb.unsimulated_catalog([adsorbate],
                                         calc_settings=calc_settings,
                                         max_atoms=max_atoms)
    # Find a list of the simulations that we have done, but only on the adsorbate
    # we're trying to match to, `matching_docs`
    with gasdb.get_adsorption_client() as ads_client:
        matching_docs = gasdb.get_docs(ads_client, 'adsorption',
                                       calc_settings=calc_settings,
                                       fingerprints=defaults.fingerprints(),
                                       adsorbates=[matching_ads])

    # Do some hashing so that we can start filtering
    cat_hashes = gasdb.hash_docs(cat_docs, ignore_ads=True)
    matching_hashes = gasdb.hash_docs(matching_docs, ignore_ads=True)
    # Filter our list of possible simulations by including them
    # only if they're in `matching_docs`
    docs = []
    for i, cat_hash in enumerate(cat_hashes.keys()):
        if cat_hash in matching_hashes:
            docs.append(cat_docs[i])

    # Post-process the docs and make the parameters list
    parameters_list = _make_parameters_list(docs, adsorbate,
                                            prioritization='random',
                                            max_predictions=max_predictions,
                                            calc_settings=calc_settings,
                                            max_atoms=max_atoms)
    return parameters_list


def from_predictions(adsorbate, prediction_min, prediction_target, prediction_max,
                     pkl=None, block=(None,), calc_settings='rpbe', max_predictions=20,
                     prioritization='gaussian', n_sigmas=6., fingerprints=None, max_atoms=50):
    '''
    Special input:
        prediction_min      The lower-bound of the prediction window that we want to hit
        prediction_target   The exact point in the prediction window that we want to hit
        prediction_max      The upper-bound of the prediction window that we want to hit
        pkl                 The location of the pickled predictions. See `gaspy_regress.predict`
        block               The block of the model that we want to use to make predictions with
        prioritization      A string that we pass to the `_make_parameters_list` function.
                            Reference that function for more details.
        n_sigmas            A float that we pass to the `_make_parameters_list` function.
                            Reference that function for more details.
        fingerprints        A dictionary that we pass to the `_make_parameters_list` function.
                            Reference that function for more details.
    Output:
        parameters_list     A list of `parameters` dictionaries that we may pass
                            to GASpy
    '''
    # Pull and unpack the predictions
    with open(pkl, 'rb') as f:
        data_ball = pickle.load(f)
    sim_data, unsim_data = data_ball
    docs, data = zip(*unsim_data)
    x_data, y_data = zip(*data)
    x, x_u = zip(*x_data)
    predictions = x

    # Trim the mongo documents and the predictions according to our prediction boundaries
    prediction_mask = (-(prediction_min < np.array(predictions)) -
                       (np.array(predictions) < prediction_max))
    docs = [docs[i] for i in np.where(prediction_mask)[0].tolist()]
    predictions = [predictions[i] for i in np.where(prediction_mask)[0].tolist()]

    # Post-process the docs and make the parameters list
    parameters_list = _make_parameters_list(docs, adsorbate,
                                            prioritization=prioritization,
                                            max_predictions=max_predictions,
                                            calc_settings=calc_settings,
                                            max_atoms=max_atoms,
                                            target=prediction_target,
                                            values=predictions,
                                            n_sigmas=n_sigmas)
    return parameters_list


def _make_parameters_list(docs, adsorbate, prioritization, max_predictions=20,
                          calc_settings='rpbe', max_atoms=50, target=None, values=None, n_sigmas=6.):
    '''
    Given a list of mongo doc dictionaries, this method will decide which of those
    docs to convert into `parameters` dictionaries for further processing.
    We do this in two steps:  1) choose and use a prioritization method
    (i.e., how to pick the docs), and then 2) trim the docs down to the number of
    simulations we want.

    Inputs:
        docs            A list of mongo doc dictionaries
        adsorbate       A string indicating the adsorbate that you want to make a
                        prediction for.
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
                        by `n_sigmas`. A higher `n_sigmas` yields a more
                        narrow selection, while a lower `n_sigmas` yields
                        a wider selection.
    Output:
        parameters_list The list of parameters dictionaries that may be sent
                        to GASpy
    '''
    # TODO:  Remove the divisor when we figure out how to keep top/bottom consistent
    if len(docs) <= max_predictions / 2:
        '''
        If we have less choices than the max number of predictions, then
        just move on. We divide by two because we're currently submitting top+bottom
        '''

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
        docs = _trim(docs, max_predictions)

    elif prioritization == 'random':
        '''
        A 'random' prioritization means that we're just going to pick things at random.
        '''
        random.shuffle(docs)
        docs = _trim(docs, max_predictions)

    elif prioritization == 'none':
        '''
        A 'none' prioritization means that we just use the docs in the order that they came
        assuming there is already some method in how they were chosen
        '''
        docs = _trim(docs, max_predictions)

    elif prioritization == 'CoordinationLength':
        '''
        A 'CoordinationLength' prioritization attempts to choose sites to try based on the number of
        elements in the coordination fingerprint. This focuses resources on small coordination sites
        which are more likely to be evenly distributed among all elements
        '''
        def get_len_coordination(doc):
            ''' Simple function to get the number of elements in the coordination FP '''
            try:
                formula = re.findall(r'([A-Z][a-z]*)(\d*)', doc['formula'])
                return [len(doc['coordination'].split('-')),
                        np.sum([int(a[1]) if a[1] != '' else 0 for a in formula])]
            except KeyError:
                return 0

        # Sort the submissions by the len of the coordination field,
        # to target small coordinations first
        random.shuffle(docs)
        all_coordination_len = utils.multimap(get_len_coordination, docs)
        sorted_coords, sorted_inds = zip(*sorted(zip(all_coordination_len, range(len(docs)))))
        docs = [docs[i] for i in sorted_inds]
        unique_coord, unique_indices = np.unique([doc['coordination'] for doc in docs],
                                                 return_index=True)
        unique_indices = sorted(unique_indices, key=lambda x: sorted_coords[x])
        docs = [docs[i] for i in unique_indices]

        print([docs[i] for i in range(3)])
        docs = _trim(docs, max_predictions)

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
            # TODO:  Remove the divisor when we figure out how to keep top/bottom consistent
            target = (max(values)-min(values))/2.
        # `dist` is the distribution we use to choose our samples, and `pdf_eval` is a
        # list of probability density values for each of the energies. Google "probability
        # density functions" if you don't know how this works.
        dist = sp.stats.norm(target, (max(values)-min(values))/n_sigmas)
        pdf_eval = map(dist.pdf, values)
        # We use np.random.choice to do the choosing. But this function needs `p`, which
        # needs to sum to one. So we re-scale pdf_eval such that its sum equals 1; rename
        # it p, and call np.random.choice
        # TODO:  Get rid of the `/2` here if we ever address the top/bottom issue
        p = (pdf_eval/sum(pdf_eval)).tolist()
        docs = np.random.choice(docs, size=max_predictions/2, replace=False, p=p)

    else:
        raise Exception('User did not provide a valid prioritization')

    # Now create the parameters list from the trimmed and processed `docs`
    parameters_list = []
    for doc in docs:
        # Define the adsorption parameters via `defaults`.
        adsorption_parameters = defaults.adsorption_parameters(adsorbate=adsorbate,
                                                               settings=calc_settings)
        # Change the fingerprint to match the coordination of the doc we are looking at.
        # Since there is a chance the user may have omitted any of these fingerprints,
        # we use EAFP to define them.
        fp = OrderedDict()
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
        # TODO:  Get rid of this loop if we ever address the top/bottom issue
        for top in [True, False]:
            slab_parameters = defaults.slab_parameters(miller=doc['miller'],
                                                       top=top,
                                                       shift=doc['shift'],
                                                       settings=calc_settings)
            # Finally:  Create the new parameters
            parameters_list.append(OrderedDict(bulk=defaults.bulk_parameters(doc['mpid'],
                                                                             settings=calc_settings,
                                                                             max_atoms=max_atoms),
                                               gas=defaults.gas_parameters(adsorbate,
                                                                           settings=calc_settings),
                                               slab=slab_parameters,
                                               adsorption=adsorption_parameters))
    return parameters_list


def _trim(_list, max_predictions):
    '''
    Trim an iterable down according to this function's `max_predictions` argument.
    Since we trim the end of the iterable, we are implicitly prioritizing the
    elements in the beginning of the list.
    '''
    # Treat max_predictions == 0 as no limit
    if max_predictions == 0:
        pass
    # TODO:  Address this if we ever address the top/bottom issue
    # We trim to half of max_predictions right now, because _make_parameters_list
    # currently creates two sets of parameters per system (i.e., top and bottom).
    # It's set up like this right now because our catalog is
    # not good at keeping track of top and bottom, so we do both (for now).
    else:
        __list = _list[:int(max_predictions/2)]
    return __list
