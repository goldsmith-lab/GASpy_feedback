# GASpy Feedback

[GASpy](https://github.com/ulissigroup/GASpy/tree/v0.1) is able to create
various catalyst-adsorbate systems and then use DFT to simulate the adsorption
energies of these systems.
[GASpy_regressions](https://github.com/ktran9891/GASpy_regressions) analyzes
GASpy's results to create surrogate models that can make predictions on DFT
calculations that we have not yet performed. This repository, which is meant to
be a submodule of GASpy, uses the predictions created by GASpy_regressions to
decide (and queue) which simulations that GASpy should perform next.

The main thing that this repository does is
[here](https://github.com/ulissigroup/GASpy_feedback/blob/master/examples/active_learning_feedback.py).
This script references a pre-determined catalog of adsorption sites that we
[enumerated](https://github.com/ulissigroup/GASpy/blob/master/examples/enumerate_dft_catalog_manually.py)
with GASpy; each of these sites has an associated adsorption energy prediction
that is created by these
[notebooks](https://github.com/ulissigroup/GASpy_regressions/blob/master/notebooks/).
These predictions are then used to determine which of the sites in our catalog
to simulate with DFT. We run this script via
[cron](https://en.wikipedia.org/wiki/Cron) to keep jobs running continuously.

# Installation

You will need to first install GASpy. Then to use GASpy_feedback, you will need
to make sure that this repository is cloned into your local repository of GASpy
as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Then run
via Docker, e.g. `docker run -v "/local/path/to/GASpy:/home/GASpy"
ulissigroup/gaspy_feedback:latest foo`.

# Reference

[Active learning across intermetallics to guide discovery of electrocatalysts
for CO2 reduction and H2
evolution](https://www.nature.com/articles/s41929-018-0142-1). Note that the
repository which we reference in this paper is version 0.1 of GASpy_feedback,
which can stil be found
[here](https://github.com/ulissigroup/GASpy_feedback/tree/v0.1).

# Versions

Current GASpy_feedback version: 0.20

For an up-to-date list of our software dependencies, you can simply check out
how we build our docker image
[here](https://github.com/ulissigroup/GASpy_feedback/blob/master/dockerfile/Dockerfile).
