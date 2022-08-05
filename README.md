# Bayesian calibration of a flood simulator using binary satellite observations

Repository for the code and datasets included in the paper "Bayesian calibration of a flood simulator using binary satellite observations" (link). The scripts were done in Python and R (they both do the same), although the Python code has a bit more complete set of functionalities.

For further inquiries: mabalbi@fi.uba.ar

Observations:

- Two binary raster images are available. The original flood extent observation "BuscotFlood92.tiff", and a masked version "BuscotFlood92_0.tiff", that masks out certain regions of the extent to reduce number of observations.

Python:

- BuscotGLUE.py: this script runs the Lisflood-fp for a square grid of roughness parameters, defined by the user.
- BuscotBC.py: this scripts performs adaptive MCMC using the Binary-Channel likelihood function. User defines priors of parameters, and parameters for the chains.
- BuscotProbit.py: this scripts performs adaptive MCMC using the multivariate probit likelihood function. User defines priors of parameters, and parameters for the chains (very similar to BuscotBC.py).
- LisfloodGP.py: repository of functions to calculate likelihoods, predictive sampling, running lisflood, and other auxiliary functions.
- mh_posterior.py: repository of functions to perform adaptive MCMC simulations
- preprocess.py: repository of functions to create Lisflood-fp input files from user defined-input
- minimax_tilting_sampler.py: auxiliary function to sample truncated multivariate normal variables.

R:

- BuscotProbit.r: All necessary functions and script to do adaptive MCMC on Lisflood-fp simulations, using probit multivariate likelihood. No predictive sampling or MCMC diagnostics included (only on Python)

Some considerations:

- All the lisflood files, data folder, and R/Python (whichever you use) files should be in the same folder to run.

- The scripts save outputs of the Lisflood-fp runs in a user-defined folder, inside a 'results' folder


