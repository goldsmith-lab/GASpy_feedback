'''
This package is meant to be a submodule to GASpy. It creates specific targets for
GASpy to simulate. This is the "active learning" part of the whole GASpy workflow.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


# from . import create_parameters

# Luigi cannot handle modules that have relative imports, which means that
# task-containing modules cannot be part of packages. Do not try to add them
# to __init__.py, because that will effectively make that module import itself,
# which creates redundancy of tasks when using Luigi.
# from . import feedback
