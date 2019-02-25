'''
This package is meant to be a submodule to GASpy. It creates specific targets for
GASpy to simulate. This is the "active learning" part of the whole GASpy workflow.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

# flake8: noqa

from .core import (get_n_jobs_to_submit,
                   randomly,
                   low_cov_ads_energies_with_guassian_noise,
                   orr_sites_with_gaussian_noise)
