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
Output:
    parameters_list A list of `parameters` dictionaries that can be
                    passed to GASpy to execute a simulation
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
sys.path.insert(0, '../')
from gaspy import defaults
from gaspy import utils
sys.path.insert(0, '../GASpy_regressions')
import regressor


def randomly(adsorbate, calc_settings='rpbe', max_predictions=20):
    ''' Call this method if you want n=`max_predictions` completely random things '''
    docs, _ = _filter_catalog(adsorbate, calc_settings=calc_settings)
    parameters_list = _make_parameters_list(docs, adsorbate,
                                            prioritization='random',
                                            max_predictions=max_predictions,
                                            calc_settings=calc_settings)
    return parameters_list


def from_matching_ads(adsorbate, matching_ads, calc_settings='rpbe', max_predictions=20):
    '''
    Call this method if you want n=`max_predictions` random sites that have already been
    relaxed with `adsorbate` on top. This method is useful for comparing a new adsorbate
    to an old one.

    Special input:
        matching_ads    The adsorbate that you want to compare to.
    '''
    # Find a list of the simulations that we haven't done yet, `cat_docs`
    cat_docs, _ = _filter_catalog(adsorbate, calc_settings=calc_settings)
    # Find a list of the simulations that we have done, but only on the adsorbate
    # we're trying to match to, `matching_docs`
    with utils.get_adsorption_db() as ads_client:
        matching_docs, _ = utils.get_docs(ads_client, 'adsorption',
                                          calc_settings=calc_settings,
                                          fingerprints=defaults.fingerprints(),
                                          adsorbates=[matching_ads])

    # Do some hashing so that we can start filtering
    cat_hashes = _hash_docs(cat_docs, ignore_ads=True)
    matching_hashes = _hash_docs(matching_docs, ignore_ads=True)
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
                                            calc_settings=calc_settings)
    return parameters_list


def from_predictions(adsorbate, prediction_min, prediction_target, prediction_max,
                     pkl=None, block='no_block',
                     calc_settings='rpbe', max_predictions=20,
                     prioritization='gaussian', n_sigmas=6.,
                     fingerprints=None):
    # pylint: disable=too-many-arguments
    '''
    Input:
        prediction_min      The lower-bound of the prediction window that we want to hit
        prediction_target   The exact point in the prediction window that we want to hit
        prediction_max      The upper-bound of the prediction window that we want to hit
        pkl                 The location of the GASpyRegressor instance that we want to
                            make predictions with
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
    # Load the catalog data
    docs, p_docs = _filter_catalog(adsorbate,
                                   calc_settings=calc_settings,
                                   fingerprints=fingerprints)

    # Load the model
    with open(pkl, 'rb') as f:
        regressor = pickle.load(f)
    # Catalog documents don't have any information about adsorbates. But if our
    # model requires information about adsorbates, then we probably need to put
    # it in.
    if 'ads' in regressor.features:
        p_docs['adsorbates'] = [[adsorbate]]*len(docs)
    # Make the predictions
    predictions = regressor.predict(p_docs, block)

    # Trim the mongo documents and the predictions according to our prediction boundaries
    prediction_mask = (-(prediction_min < np.array(predictions)) - \
                        (np.array(predictions) < prediction_max))
    docs = [docs[i] for i in np.where(prediction_mask)[0].tolist()]
    predictions = [predictions[i] for i in np.where(prediction_mask)[0].tolist()]

    # Post-process the docs and make the parameters list
    parameters_list = _make_parameters_list(docs, adsorbate,
                                            prioritization=prioritization,
                                            max_predictions=max_predictions,
                                            calc_settings=calc_settings,
                                            target=prediction_target,
                                            values=predictions,
                                            n_sigmas=n_sigmas)
    return parameters_list


def _make_parameters_list(docs, adsorbate, prioritization, max_predictions=20,
                          calc_settings='rpbe', target=None, values=None, n_sigmas=6.):
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
    # pylint: disable=too-many-branches, too-many-arguments
    # TODO:  Remove the divisor when we figure out how to keep top/bottom consistent
    if len(docs) <= max_predictions/2:
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
        p = (pdf_eval/sum(pdf_eval)).tolist()
        docs = np.random.choice(docs, size=max_predictions/2, replace=False, p=p) # pylint: disable=no-member

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
                                                       settings=calc_settings)
            # Finally:  Create the new parameters
            parameters_list.append({'bulk': defaults.bulk_parameters(doc['mpid'], settings=calc_settings),
                                    'gas': defaults.gas_parameters(adsorbate, settings=calc_settings),
                                    'slab': slab_parameters,
                                    'adsorption': adsorption_parameters})
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


def _filter_catalog(adsorbate, calc_settings='rpbe', fingerprints=None):
    '''
    Pull out all of the items in the catalog, and filter out all of the items
    that we have already simulated.

    Inputs:
        adsorbate       A string indicating the adsorbate that you want to make a
                        prediction for.
        calc_settings   The calculation settings that we want to use. If we are using
                        something other than beef-vdw or rpbe, then we need to do some
                        more hard-coding here so that we know what in the catalog
                        can work as a flag for this new calculation method.
        fingerprints    A dictionary of fingerprints and their locations in our
                        mongo documents. This is how we can pull out more (or less)
                        information from our database.
    Output:
        docs    A list of dictionaries for various fingerprints. Useful for
                creating lists of GASpy `parameters` dictionaries.
        p_docs  A dictionary of lists for various fingerprints. Useful for
                passing to GASpyRegressor.predict
    '''
    # Default value for fingerprints. Since it's a mutable dictionary, we define it
    # down here instead of in the __init__ line.
    if not fingerprints:
        fingerprints = defaults.fingerprints()

    # Fetch mongo docs for our results and catalog databases so that we can
    # start filtering out cataloged sites that we've already simulated.
    with utils.get_adsorption_db() as ads_client:
        ads_docs, _ = utils.get_docs(ads_client, 'adsorption',
                                     calc_settings=calc_settings,
                                     fingerprints=fingerprints,
                                     adsorbates=[adsorbate])
    with utils.get_catalog_db() as cat_client:
        cat_docs, cat_p_docs = utils.get_docs(cat_client, 'catalog', fingerprints=fingerprints)
    # Hash the docs so that we can filter out any items in the catalog
    # that we have already relaxed. Note that we keep `ads_hash` in a dict so
    # that we can search through it, but we turn `cat_hash` into a list so that
    # we can iterate through it alongside `cat_docs`
    ads_hashes = _hash_docs(ads_docs)
    cat_hashes = _hash_docs(cat_docs).keys()

    # Perform the filtering while simultaneously populating the `docs` output
    docs = [cat_docs[i]
            for i, cat_hash in enumerate(cat_hashes)
            if cat_hash not in ads_hashes]
    # Do the same for the `p_docs` output
    p_docs = dict.fromkeys(cat_p_docs)
    for fingerprint, data in cat_p_docs.iteritems():
        p_docs[fingerprint] = [data[i]
                               for i, cat_hash in enumerate(cat_hashes)
                               if cat_hash not in ads_hashes]

    return docs, p_docs


def _hash_docs(docs, ignore_ads=False):
    '''
    This function helps convert the important characteristics of our systems into hashes
    so that we may sort through them more quickly. This is important to do when trying to
    compare entries in our two databases; it helps speed things up.

    Input:
        docs        Mongo docs (list of dictionaries) that have been created using the
                    gaspy.utils.get_docs function. Note that this is the unparsed version
                    of mongo documents.
        ignore_ads  A boolean that decides whether or not we hash the adsorbate.
                    This is useful mainly for the "matching_ads" function.
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
            # Ignore mongo ID, because that'll always cause things to hash differently
            if key != 'mongo_id':
                # Ignore adsorbates if the user wants to, as per the argument
                if not (ignore_ads and key == 'adsorbate_names'):
                    # Note that we turn the values into strings explicitly, because some
                    # fingerprint features may not be strings (e.g., list of miller indices).
                    system += str(key + '=' + str(value) + '; ')
        systems[hash(system)] = None

    return systems
