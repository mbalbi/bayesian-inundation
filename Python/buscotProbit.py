import numpy as np
import scipy.stats as st
import os

from LisfloodGP import logPosterior, logLikelihoodProbit

from mh_posterior import mh_paths_sampler
from mh_posterior import paths_diagnostics

import osgeo.gdal as gdal

## ===========================================================================

bounds = [22,45,0,30]

# Data (binary flood extent)
observed = 'Observations//BuscotFlood92.tiff'
src = gdal.Open( observed )
z = src.GetRasterBand(1).ReadAsArray()
z = z[bounds[0]:bounds[1],bounds[2]:bounds[3]]
src = None

# Spatial locations
i_coords, j_coords = np.meshgrid( range(z.shape[0]), range(z.shape[1]),
                                  indexing='ij')
i_coords = i_coords.flatten()
j_coords = j_coords.flatten()
x = np.vstack( (i_coords,j_coords) ).T
z_flat = z.flatten()

# Prior of model parameters
vars = ['r_ch', 'r_fp', 'logt1', 'logt2', 'logse']
logrch_prior = st.norm( loc=np.log(0.03), scale=0.6 )
logrfp_prior = st.norm( loc=np.log(0.09), scale=0.3 )
logt1_prior = st.norm( loc=0, scale=0.1 )
logt2_prior = st.norm( loc=-3, scale=1 )
logse_prior = st.norm( loc=0, scale=1 )

logpriors = [logrch_prior, logrfp_prior, logt1_prior, logt2_prior, logse_prior]

## MCMC
outputDir = 'results//Probit_1410_U2'
mode = 'normal'
tune = 2000
tune_intvl = 60 
resultscsv = os.path.join( outputDir, 'mh_summary.csv' )
if not os.path.isdir(outputDir):
    os.mkdir(outputDir)
    
target_logpdf = lambda params, output: logPosterior( x, z_flat, logpriors,
                                                     logLikelihoodProbit,
                                                     params, output )

# Jump distribution 
sigmas = np.array([0.03, 0.03, 0.01, 0.01,0.01])
cov = sigmas**2*np.eye(len(sigmas))

# Metadata file
md_file = os.path.join( outputDir, 'metadata.txt' )
f = open( md_file, "w" )
f.write( "Mode: " + mode + ' (tune: ' + str(tune) + ', intvl: ' + str(tune_intvl) + ')\n')
f.write( "Initial COV: " + str(sigmas) + '\n' )
f.write( "LogPriors: \n" )
for lp in logpriors:
    f.write( lp.dist.name + '\t' + str(lp.kwds) + '\n')
f.close()

# MH Sampler
Nsim = 10000
burnin = int(Nsim/2)
Npaths = 2
thin = 1
x0 = np.array([x.rvs( size=Npaths+1 ) for x in logpriors]).T
Xbin, Xstack, acceptance, diagnostics = mh_paths_sampler(vars, target_logpdf, 
                                                    Npaths, Nsim, x0, cov,
                                                    burnin=burnin, thin=thin,
                                                    sampler=mode, tune=tune,
                                                    tune_intvl=tune_intvl,
                                                    outputDir=outputDir)

# MCMC Diagnostics
R, var_j, rhot, neff = paths_diagnostics( Xbin, True, *logpriors )
