# GASpy Feedback

## Purpose
[GASpy](https://github.com/ktran9891/GASpy) is able to create various catalyst-adsorbate systems
and then use DFT to simulate the adsorption energies of these systems.
[GASpy_regressions](https://github.com/ktran9891/GASpy_regressions) analyzes GASpy's results to
create surrogate models that can identify potentially high-performing catalysts. This repository,
which is meant to be a submodule of GASpy, uses the models created by GASpy_regressions to decide
which simulations that GASpy should perform next.

## Overview
`gaspy_feedback` is a Python package that contains the `create_parameters.py` module, which
contains functions to creates list of `parameters` for GASpy to perform simulations. This list
is created by coupling models created by GASpy_regressions with the adsorption site catalog
created by GASpy to create a priority list of different adsorption sites to simulate. This is
the "active learning" part of the workflow.

This list is used by `feedback.py`, which contains [Luigi](https://github.com/spotify/luigi)
tasks that tell GASpy to perform simulations of the systems which we think might perform the best.

`queue_predicted.sh` is a template bash script to perform the feedback simulations.

## Installation
Remember to add the repo to your Python path. The module importing assumes that you have GASpy in your Python path.
You can do so by putting the following in your `.bashrc`:
```
export PYTHONPATH="/path/to/GASpy/GASpy_feedback:${PYTHONPATH}"
```
