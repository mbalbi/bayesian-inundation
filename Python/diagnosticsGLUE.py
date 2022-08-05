import csv
import os
import numpy as np
import re, ast
import matplotlib.cm as cm
import json

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6, 'lines.linewidth':0.8, 'axes.linewidth':0.5,
                     'xtick.major.width':0.5, 'ytick.major.width':0.5})

import scipy.stats as st

from mh_posterior import paths_diagnostics

from LisfloodGP import predGP, predProbitM0, readPickle, predProbit

import scipy.stats.kde as kde
import seaborn as sns

import osgeo.gdal as gdal

import pickle

import pandas as pd

def read_mcmc_results( folder, burnin, nvars ):
    
    # Read paths from folder
    path_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".csv"):
                path_files.append( os.path.join(root, file) )
            if file.endswith(".txt"):
                mdata_file = os.path.join(root, file)

    # Number of sims per path
    csvfile = open(path_files[0], "r")
    reader = csv.reader(csvfile, delimiter=',')
    Nsim = sum(1 for row in reader) - 2 # 1 for header and 1 for 1st iter
    csvfile.seek(0)

    # Read 
    j = 0
    Xpaths = np.zeros([Nsim,nvars,len(path_files)])
    logposterior = np.zeros([Nsim,len(path_files)])
    maxfiles = []
    for filename in path_files:
        
        csvfile = open(filename, "r")
        reader = csv.reader(csvfile, delimiter=',')

        # Header
        header = next(reader)
        
        # First iteration
        row0 = next(reader)
        Xprev = [float(a) for a in row0[:nvars]]
        logposterior_prev = row0[nvars+2]
        maxfilesprev = row0[-1]
        
        i = 0
        for row in reader:
            acc = int( float( row[nvars+1] ) )
            if acc == 1:
                Xpaths[i,:,j] = [float(a) for a in row[:nvars]]
                logposterior[i,j] = row[nvars+2]
                maxfiles.append( row[-1] )
            else: # append last ones
                Xpaths[i,:,j] = Xprev
                logposterior[i,j] = logposterior_prev
                maxfiles.append( maxfilesprev )
            # Update
            Xprev = Xpaths[i,:,j]
            logposterior_prev = logposterior[i,j]
            maxfilesprev = maxfiles[-1]
            i += 1
        
        # Update path number
        j += 1
    
    # Stack paths
    Nperpath = Xpaths.shape[0]
    Xbin = Xpaths[burnin:,:,:]
    Xstack = Xbin[:,:,0]
    logposteriorbin = logposterior[burnin:,:]
    logposteriorstack = logposteriorbin[:,0]
    maxfiles_bin = maxfiles[ burnin:Nperpath ]
    for i in range( 1, Xbin.shape[2] ):
        Xstack = np.vstack( [Xstack, Xbin[:,:,i]] )
        logposteriorstack = np.hstack( [logposteriorstack, logposteriorbin[:,i]] )
        maxfiles_bin += maxfiles[ i*Nperpath + burnin : (i+1)*Nperpath ]
        
    # Read metadata (create logpriors)
    mdatafile = open( mdata_file, "r" )
    lines = mdatafile.readlines()
    logpriors = []
    for line in lines[3:-2]:
        aux = re.split('\t|\n',line)[:-1]
        dist_name = aux[0]
        # params = [json.loads(aux[i]) for i in range(1,len(aux))]
        params = {'loc':float(aux[1]), 'scale':float(aux[2])}
        # params = ast.literal_eval( aux[1] )
        logpriors.append( getattr( st, 'norm' )(**params) )
      
    return Xpaths, Xstack, logpriors, maxfiles_bin, logposteriorstack

def predictiveModel( Xstack, x, N, z, maxfiles, bounds=None, folder='' ):
      
    # Prediciton
    ies = np.random.choice(np.arange(0,Xstack.shape[0],1), N, replace=False)

    S = np.zeros( [N, z.shape[0], z.shape[1] ] ) # Only lisflood model
    inad = np.zeros( [N, z.shape[0], z.shape[1] ] ) # Lisflood plus inadequacy mean
    fs = np.zeros( [N, z.shape[0], z.shape[1] ] ) # Normal sample

    for i in range(N):
        print('predictive {} of {}'.format(i,N))
        # Load parameters
        params_aux = Xstack[ ies[i] ]
        params_log = np.array( params_aux )
        params = np.exp( params_log )

        # Cond Mean and covariance
        filename = os.path.basename( maxfiles[ies[i]] )
        filepath = os.path.join( folder, filename )
        inad_mean, fcov, S[i] = predProbit( x, x, z, params, 
                                            maxfile=filepath, bounds=bounds)
        
        # Sample
        inadi = np.random.multivariate_normal( mean=inad_mean,
                                               cov=fcov,
                                               check_valid='ignore' )
        # inadi[ fsi<0.01 ] = 0
        # inadi = np.maximum( inadi, -S[i].flatten() )
        
        # Rebuild grid
        # inadi_aux = np.zeros( zflat.shape )
        # inadi_aux[ obs_index ] = inadi
        inad[i, x[:,0], x[:,1]] = inadi
        fsi = inad[i] + S[i]
        fsi[ fsi<0 ] = 0
        fs[i] = fsi

    return fs, inad, S

def indexMap():
    return 0


## =============================================================================

# Read MCMC
folder = '/...'


Xpaths, Xstack, logpriors, maxfiles, logp = read_mcmc_results( folder, burnin=5000,
                                                               nvars=5 )

# Compute & Plot diagnostics
R, var_j, rhot, neff = paths_diagnostics( Xpaths[5000:,:,:], True, *logpriors )
plt.show()

# Training Data
bounds = [9,45,0,68]
# bounds = [0,48,0,76]
src = gdal.Open( 'Observations//BuscotFlood92_0.tiff' )
z = src.GetRasterBand(1).ReadAsArray()
src = None
z = z[bounds[0]:bounds[1],bounds[2]:bounds[3]]
# DEM
dem = 'Buscot.dem.asc'
src = gdal.Open( dem )
dem = src.GetRasterBand(1).ReadAsArray()
src = None
# Spatial locations
zflat = z.flatten()
i_coords, j_coords = np.where( z >= 0 )
x = np.vstack( (i_coords,j_coords) ).T

# Predictions
N = 100
fs, inad, S = predictiveModel( Xstack, x, N, z, maxfiles, bounds=bounds,
                               folder=folder )

# Probability of flooding (P[h>threshold])
S0 = fs>0.02
Pmap = S0.mean(axis=0)

# Best fit
best_fit_index = np.where( logp == logp.max() )[0][0]
Xstack_best = np.array([ Xstack[best_fit_index,:] ])
fs_best, inad_best, S_best = predictiveModel( Xstack_best,
                                              x, 1, z, maxfiles, bounds=bounds,
                                              folder=folder )

# Goodness-of-fit map
gof = Pmap * ( z - 1 ) + ( 1 - Pmap ) * z

# Posterior marginal densities
names = [ r'$\log\ r_{ch}$', r'$\log\ r_{fp}$', 
          r'$\log\ \theta_1$', r'$\log\ \theta_2$', r'$\log\ s_n$' ]
fig, ax = plt.subplots( figsize=(8.3/2.54,14/2.54), ncols=1, nrows=len(logpriors) )
for i in range( len(logpriors) ):
    mu, sig = logpriors[i].stats()
    ax[i] = sns.kdeplot( Xstack[:,i] , linewidth=0, ax=ax[i], fill=True, alpha=0.5,
                         label='posterior')
    x = np.arange( mu-2.5*np.sqrt(sig), mu+2.5*np.sqrt(sig), sig/100)
    o2 = ax[i].fill_between( x, logpriors[i].pdf(x), alpha=.5, label='prior' )
    ax[i].set_xlabel( names[i] )

ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)
fig.tight_layout()
fig.subplots_adjust( hspace=0.55 )

# Plot best fit prediction
fig, ax = plt.subplots( figsize=(8.3/2.54,15/2.54), ncols=1, nrows=3 )

masked_data = np.ma.masked_where( fs_best <= 0, fs_best)
dem_img = ax[0].imshow( dem[bounds[0]:bounds[1],bounds[2]:bounds[3]], cmap=cm.gray )
cmap = cm.Blues  # Can be any colormap that you want after the cm
image = ax[0].imshow( masked_data[0,:,:], cmap=cmap, interpolation='none',
                      vmin=0, vmax=3.5)
cbar = fig.colorbar( image, ax=ax[0] )
cbar.set_label('Flood depth [m]')
ax[0].contour( z, levels=[.9999, 1.0001], colors='k', linestyles='--' )
ax[0].set_xlabel( 'x distance [km]' )
ax[0].set_ylabel( 'y distance [km]' )

masked_data = np.ma.masked_where( S_best <= 0, S_best)
dem_img = ax[1].imshow( dem[bounds[0]:bounds[1],bounds[2]:bounds[3]], cmap=cm.gray )
cmap = cm.Blues  # Can be any colormap that you want after the cm
image = ax[1].imshow( masked_data[0,:,:], cmap=cmap, interpolation='none',
                     vmin=0, vmax=3.5)
cbar = fig.colorbar( image, ax=ax[1] )
cbar.set_label('Flood depth [m]')
ax[1].contour( z, levels=[.9999, 1.0001], colors='k', linestyles='--' )
ax[1].set_xlabel( 'x distance [km]' )
ax[1].set_ylabel( 'y distance [km]' )

masked_data = np.ma.masked_where( inad_best == 0, inad_best)
dem_img = ax[2].imshow( dem[bounds[0]:bounds[1],bounds[2]:bounds[3]], cmap=cm.gray )
cmap = cm.Blues  # Can be any colormap that you want after the cm
image = ax[2].imshow( masked_data[0,:,:], cmap=cmap, interpolation='none' )
cbar = fig.colorbar( image, ax=ax[2] )
cbar.set_label('Inadequacy [m]')
ax[2].contour( z, levels=[.9999, 1.0001], colors='k', linestyles='--' )
ax[2].set_xlabel( 'x distance [km]' )
ax[2].set_ylabel( 'y distance [km]' )

fig.tight_layout()
plt.show()

# Joint Posterior
Z_obs = z[bounds[0]:bounds[1],bounds[2]:bounds[3]]
X0 = np.exp(Xstack[:,:2])
df = pd.DataFrame( np.exp(Xstack), columns=['rch','rfp','theta1','theta2','sn'])
g = sns.JointGrid( data=df, x='rch', y='rfp', height=8.3/2.54, xlim=[0,0.15], ylim=[0,0.15] )
g.plot_joint( sns.kdeplot, fill=True, cmap='viridis' )
g.plot_marginals( sns.kdeplot, fill=True, linewidth=0.5 )
g.set_axis_labels( xlabel=r'$r_{ch}$', ylabel=r'$r_{fp}$' )
g.refline( x=0.025, y=0.035 )
g.fig.tight_layout()

# Probability of flood
fig, ax = plt.subplots( figsize=(8.3/2.54,10/2.54), ncols=1, nrows=2 )
o3 = ax[0].imshow( Pmap, extent=[0,30*0.05,0,23*0.05] )
ax[0].contour( z[::-1,:],
               levels=[.9999, 1.0001], colors='k', linestyles='--',
               extent=[0,30*0.05,0,23*0.05])
ax[0].set_xlabel( 'x distance [km]' )
ax[0].set_ylabel( 'y distance [km]' )
bar3 = fig.colorbar( o3, ax=ax[0] )
bar3.set_label('probability of flood')

# Goodness-of-fit
o4 = ax[1].imshow( gof,
                   extent=[0,30*0.05,0,23*0.05], cmap='bwr',
                   vmin=-1, vmax=1)
ax[1].contour( z[::-1,:], 
               levels=[.9999, 1.0001], colors='k', linestyles='--',
               extent=[0,30*0.05,0,23*0.05] )
ax[1].set_xlabel( 'x distance [km]' )
ax[1].set_ylabel( 'y distance [km]' )
bar4 = fig.colorbar( o4, ax=ax[1], ticks=np.arange(-1,1,0.5) )
bar4.set_label(r'$\leftarrow$ underpred   $\rho$   overpred $\rightarrow$')

# Settings
fig.tight_layout()

plt.show()
