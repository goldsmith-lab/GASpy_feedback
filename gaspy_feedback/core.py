'''
This submodule contains various functions that are meant to be used to create
gaspy tasks to run in a continuous/automatic fashion for active optimization.

Note that these functions will only give you the tasks. You will have to
execute them yourself using either `gaspy.tasks.run_task` or
`gaspy_tasks.schedule_tasks`.
'''

__authors__ = ['Kevin Tran', 'Zachary W. Ulissi']
__emails__ = ['ktran@andrew.cmu.edu', 'zulissi@andrew.cmu.edu']

import numpy as np
from scipy.stats import norm
from gaspy import defaults
from gaspy.gasdb import (get_low_coverage_docs,
                         get_catalog_docs_with_predictions,
                         get_unsimulated_catalog_docs,
                         _get_attempted_adsorption_docs)
from gaspy.tasks.metadata_calculators import CalculateAdsorptionEnergy


def randomly(adsorbate, n_calcs=50, max_atoms=80, vasp_settings=None):
    '''
    This function will pick random, unsimulated sites from our catalog and then
    sumbit adsorption energy calculations.

    Args:
        adsorbate       A string indicating the adsorbate that you want to
                        calculate an adsorption energy for. It should probably
                        be one of the adsorbates in
                        `gaspy.defaults.adsorbates()`.
        n_calcs         A positive integer indicating how many adsorption
                        energy calculations you want GASpy to perform.
        max_atoms       A positive integer indicating the maximum number of
                        atoms that you want in the calculations you want to
                        perform. A higher `max_atoms` allows for longer-running
                        calculations, and vice versa.
        vasp_settings   [optional] An OrderedDict containing the default energy
                        cutoff, VASP pseudo-potential version number
                        (pp_version), and exchange-correlational settings. This
                        should be obtained (and modified, if necessary) from
                        `gaspy.defaults.adslab_settings()['vasp']`. If `None`,
                        then pulls default settings.
    Returns:
        tasks   A list of the `CalculateAdsorptionEnergy` tasks that we chose
    '''
    # Python doesn't like mutable default arguments
    if vasp_settings is None:
        vasp_settings = defaults.adslab_settings()['vasp']

    # Find unsimulated sites, take out ones that are too big, then pick some at
    # random.
    catalog_docs = get_unsimulated_catalog_docs(adsorbate, vasp_settings=vasp_settings)
    catalog_docs = [doc for doc in catalog_docs if doc['natoms'] <= max_atoms]
    docs_to_run = np.random.choice(catalog_docs, size=n_calcs, replace=False)

    # Parse the catalog documents into calculation tasks
    tasks = []
    for doc in docs_to_run:
        task = CalculateAdsorptionEnergy(adsorbate_name=adsorbate,
                                         adsorption_site=doc['adsorption_site'],
                                         mpid=doc['mpid'],
                                         miller_indices=doc['miller'],
                                         shift=doc['shift'],
                                         top=doc['top'],
                                         adslab_vasp_settings=vasp_settings)
        tasks.append(task)
    return tasks


def low_cov_ads_energies_with_guassian_noise(adsorbate, energy_target, stdev,
                                             n_calcs=50,
                                             model_tag=defaults.model(),
                                             max_atoms=80,
                                             vasp_settings=None):
    '''
    This task function will use GASpy to calculate adsorption energies for
    various adsorption sites. We choose only sites that we predict to have the
    lowest adsorption energy on their respective surfaces (which are therefore
    the "low coverage" sites) as per both DFT and surrogate models. We also
    choose a subset of these low-coverage sites using Gaussian noise that is
    centered at the target and with a specified standard deviation.

    Args:
        adsorbate       A string indicating the adsorbate that you want to
                        calculate an adsorption energy for. It should probably
                        be one of the adsorbates in
                        `gaspy.defaults.adsorbates()`.
        energy_target   A float indicating the adsorption energy that you're
                        trying to target, in eV
        stdev           A float indicating the standard deviation of the Gaussian
                        noise you want to add to the selection.
        model_tag       A string indicating which surrogate model you want to
                        use when estimating what you think the adsorption
                        energy is going to be.
        n_calcs         A positive integer indicating how many adsorption
                        energy calculations you want GASpy to perform.
        max_atoms       A positive integer indicating the maximum number of
                        atoms that you want in the calculations you want to
                        perform. A higher `max_atoms` allows for longer-running
                        calculations, and vice versa.
        vasp_settings   [optional] An OrderedDict containing the default energy
                        cutoff, VASP pseudo-potential version number
                        (pp_version), and exchange-correlational settings. This
                        should be obtained (and modified, if necessary) from
                        `gaspy.defaults.adslab_settings()['vasp']`. If `None`,
                        then pulls default settings.
    Returns:
        tasks   A list of the `CalculateAdsorptionEnergy` tasks that we chose
    '''
    # Python doesn't like mutable default arguments
    if vasp_settings is None:
        vasp_settings = defaults.adslab_settings()['vasp']

    # Get the documents for the low-coverage sites while taking out sites that
    # are too big
    low_coverage_docs = [doc for doc in get_low_coverage_docs(adsorbate, model_tag)
                         if (doc['DFT_calulated'] is False and
                             doc['natoms'] <= max_atoms)]

    # Some calculations/sites result in an adsorbate moving so far that the
    # fingerprint changes. If we try to calculate the energy for these sites,
    # GASpy ends up trying to run the same calculation again. We avoid this by
    # making sure that the site we're trying to submit here doesn't match with
    # any site that we've tried (as opposed to checking against sites that we
    # have).
    attempted_docs = _get_attempted_adsorption_docs(adsorbate, vasp_settings)
    attempted_fingerprints = set(__fingerprint_doc(doc) for doc in attempted_docs)
    unattempted_docs = []
    for doc in low_coverage_docs:
        if __fingerprint_doc(doc) not in attempted_fingerprints:
            unattempted_docs.append(doc)

    # Choose the documents with Gaussian noise
    energies = [doc['energy'] for doc in unattempted_docs]
    gaussian_distribution = norm(loc=energy_target, scale=stdev)
    probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]
    probabilities = probability_densities/sum(probability_densities)
    docs_to_run = np.random.choice(unattempted_docs, size=n_calcs, replace=False, p=probabilities)

    # Make the GASpy tasks to do the calculations
    tasks = []
    for doc in docs_to_run:
        task = CalculateAdsorptionEnergy(adsorbate_name=adsorbate,
                                         adsorption_site=doc['adsorption_site'],
                                         mpid=doc['mpid'],
                                         miller_indices=doc['miller'],
                                         shift=doc['shift'],
                                         top=doc['top'],
                                         adslab_vasp_settings=vasp_settings)
        tasks.append(task)
    return tasks


def __fingerprint_doc(doc):
    fingerprint = (doc['mpid'],
                   tuple(doc['miller']),
                   doc['shift'],
                   doc['top'],
                   doc['adsorption_site'],
                   doc['coordination'],
                   tuple(doc['neighborcoord']))
    return fingerprint


def orr_sites_with_gaussian_noise(adsorbate, orr_target, stdev,
                                  rotations=None, n_calcs=50,
                                  model_tag=defaults.model(),
                                  max_atoms=80, vasp_settings=None):
    '''
    This task function will use GASpy to calculate adsorption energies for
    various adsorption sites. We choose sites near a targeted onset potential
    using Gaussian noise that is centered at the target and with a specified
    standard deviation.

    Args:
        adsorbate       A string indicating the adsorbate that you want to
                        calculate an adsorption energy for. It should probably
                        be one of the adsorbates in
                        `gaspy.defaults.adsorbates()`.
        orr_target      A float indicating the 4e ORR onset potential that
                        you're trying to target
        stdev           A float indicating the standard deviation of the Gaussian
                        noise you want to add to the selection.
        rotations       A list containing the angles (in degrees) in which to
                        rotate the adsorbate after it is placed at the
                        adsorption site. These values will be used for 'phi' in
                        the rotation dictionary.
        model_tag       A string indicating which surrogate model you want to
                        use when estimating what you think the adsorption
                        energy is going to be.
        n_calcs         A positive integer indicating how many adsorption
                        energy calculations you want GASpy to perform.
        max_atoms       A positive integer indicating the maximum number of
                        atoms that you want in the calculations you want to
                        perform. A higher `max_atoms` allows for longer-running
                        calculations, and vice versa.
        vasp_settings   [optional] An OrderedDict containing the default energy
                        cutoff, VASP pseudo-potential version number
                        (pp_version), and exchange-correlational settings. This
                        should be obtained (and modified, if necessary) from
                        `gaspy.defaults.adslab_settings()['vasp']`. If `None`,
                        then pulls default settings.
    Returns:
        tasks   A list of the `CalculateAdsorptionEnergy` tasks that we chose
    '''
    # Python doesn't like mutable default arguments
    if rotations is None:
        rotations = [0., 90., 180., 270.]
    if vasp_settings is None:
        vasp_settings = defaults.adslab_settings()['vasp']

    # Find all of our unsimulated catalog sites
    rotation_list = [{'phi': rot, 'theta': 0., 'psi': 0.} for rot in rotations]
    unsim_cat_docs = get_unsimulated_catalog_docs(adsorbate, rotation_list)

    # Get all of our ORR predictions, then push them into our catalog of
    # unsimulated sites
    cat_docs = [doc for doc in get_catalog_docs_with_predictions()
                if doc['natoms'] <= max_atoms]
    cat_docs_by_mongo_id = {doc['mongo_id']: doc for doc in cat_docs}
    for doc in unsim_cat_docs:
        doc['predictions'] = cat_docs_by_mongo_id[doc['mongo_id']]

    # Choose the documents with Gaussian noise
    potentials = [doc['predictions']['orr_onset_potential_4e'][model_tag]
                  for doc in unsim_cat_docs]
    gaussian_distribution = norm(loc=orr_target, scale=stdev)
    probability_densities = [gaussian_distribution.pdf(potential) for potential in potentials]
    probabilities = probability_densities/sum(probability_densities)
    docs_to_run = np.random.choice(unsim_cat_docs, size=n_calcs, replace=False, p=probabilities)

    # Make the GASpy tasks to do the calculations
    tasks = []
    for doc in docs_to_run:
        task = CalculateAdsorptionEnergy(adsorbate_name=adsorbate,
                                         adsorption_site=doc['adsorption_site'],
                                         rotation=doc['adsorbate_rotation'],
                                         mpid=doc['mpid'],
                                         miller_indices=doc['miller'],
                                         shift=doc['shift'],
                                         top=doc['top'],
                                         adslab_vasp_settings=vasp_settings)
        tasks.append(task)
    return tasks
