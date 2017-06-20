'''
This script contains classes/tasks for Luigi to execute.

These tasks use the `GASPredict` class in this submodule to determine what systems
to relax next, and then it performs the relaxation.
'''
__author__ = 'Kevin Tran'
__email__ = '<ktran@andrew.cmu.edu>'


# Since this is in a submodule, we add the parent folder to the python path
import sys
sys.path.append("../..")
from collections import OrderedDict as ODict
from gaspy_toolbox import FingerprintUnrelaxedAdslabs as fp_unr_adslb
# from gaspy_toolbox import DumpToLocalDB
from gaspy_toolbox import default_calc_settings
from gaspy_toolbox import default_parameter_bulk as def_par_bulk
from gaspy_toolbox import default_parameter_gas as def_par_gas
from gaspy_toolbox import default_parameter_slab as def_par_slab
from gaspy_toolbox import default_parameter_adsorption as def_par_ads
from gaspy_toolbox import UpdateAllDB
from gaspy_toolbox import FingerprintRelaxedAdslab
from gaspy_toolbox import UpdateEnumerations
from pymatgen.core.surface import get_symmetrically_distinct_miller_indices
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.matproj.rest import MPRester
import luigi
from ase.db import connect
import numpy as np


DB_LOC = '/global/cscratch1/sd/zulissi/GASpy_DB'    # Cori
#DB_LOC = '/home-research/ktran/GASpy'               # Gilgamesh
#DB_LOC = '/Users/Ktran/Nerd/GASpy'                  # Mac

# Exchange correlational
#XC = 'rpbe'
XC = 'beef-vdw'
# MAX_MILLER applies to both enumerating or calculating adsorptions
MAX_MILLER = 3
# Elements that may be inside the materials we want to look at
WHITELIST = []


class UpdateDBsForFeedback(luigi.WrapperTask):
    '''
    This class calls on the DumpToAuxDB class in gaspy_toolbox.py so that we can
    dump the fireworks database into the Aux database. We would normally
    just use fireworks, but calling fireworks from a remote cluster is slow. So we speed
    up the calls by dumping the data to the Aux DB, where querying is fast.
    '''
    # If nowrite = luigi.BoolParameter(True), then we write to the database.
    # If nowrite = luigi.BoolParameter(False), then only FingerprintRelaxedAdslabs.
    # Note that if no argument is passed to BoolParameter, it defaults to False.
    writedb = luigi.BoolParameter(True)
    # The maximum number of rows to dump to the Local DB. Enter zero if you want no limit.
    max_processes = luigi.IntParameter(0)

    def requires(self):
        '''
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires`
        method.
        '''
        if self.writedb:
            yield UpdateAllDB(writeDB=True, max_processes=self.max_processes)
        else:
            yield UpdateAllDB(writeDB=False, max_processes=self.max_processes)


class CalculateAuAgAdsorptions(luigi.WrapperTask):
    '''
    This class is meant to be called by Luigi to begin relaxations of a particular set of
    adsorption sites.
    '''
    # Declare this task's arguments and establish defaults. They may be overridden
    # on the command line.
    xc = luigi.Parameter(XC)
    #xc = luigi.Parameter('beef-vdw')
    writeDB = luigi.BoolParameter(False)

    def requires(self):
        '''
        Luigi automatically runs the `requires` method whenever we tell it to execute a
        class. Since we are not truly setting up a dependency (i.e., setting up `requires`,
        `run`, and `output` methods), we put all of the "action" into the `requires` method.
        '''
        # Fetch the materials we want to look at
        materials = select_materials(WHITELIST, ALLOY)
        # Define the calculation settings
        calc_settings = default_calc_settings(self.xc)

        # For each coordination, submit calculations
        n_submits = 0
        for coord_num, coord in enumerate(coords):

            # There are actually a subset of systems in the enumerated_ads*.db that yield the
            # coordination we are looking at now. Here, we pull out the database row numbers
            # of this subset, as well as the number of atoms in each of these systems.
            subset_row_nums, subset_n_atoms = zip(*[[row_num, site_rows[row_num].natoms]
                                                    for row_num in range(n_rows)
                                                    if inv[row_num] == coord_num])
            # To save time, we will only submit calculations for a system in our subset which
            # has the least number of atoms.
            row = site_rows[subset_row_nums[np.argmin(subset_n_atoms)]]

            # Submit calculations for this site with the adsorbates in "ADS"
            for ads in ADS:
                # Only submit calculations if we have not yet made a calculation for this
                # coordination/adsorbate pair
                if len([result for result in enrg_rows
                        if result.adsorbate == ads
                        and (result.coordination == row.coordination
                             or result.initial_coordination == row.coordination)
                       ]
                      ) == 0:

                    # Submit for both top and bottom
                    for top in [True, False]:
                        # Dictionary management
                        ads_parameter = def_par_ads(ads)
                        ads_parameter['adsorbates'][0]['fp'] = {'coordination':row.coordination}
                        parameters = {'bulk': def_par_bulk(row.mpid),
                                      'slab': def_par_slab([int(index) for index in row.miller[1:-1].split(', ')],
                                                           top=top, shift=row.shift),
                                      'gas': def_par_gas(ads),
                                      'adsorption': ads_parameter}

                        # Submit calculations
                        yield FingerprintRelaxedAdslab(parameters=parameters)

    def run(self):
        return self.writeDB
