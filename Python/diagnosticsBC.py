import csv
import os
import numpy as np
import re, ast
import json
import matplotlib.cm as cm
import ast

import scipy.stats as st

from mh_posterior import paths_diagnostics

from LisfloodGP import predGP, predBC, readPickle

import matplotlib.pyplot as plt

import scipy.stats.kde as kde
import seaborn as sns
import pandas as pd

import pickle
import osgeo.gdal as gdal

plt.rcParams.update({'font.size': 6, 'lines.linewidth':0.8, 'axes.linewidth':0.5,
                     'xtick.major.width':0.5, 'ytick.major.width':0.5})

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
    for line in lines[3:]:
        aux = re.split('\t|\n',line)[:-1]
        dist_name = aux[0]
        # params = [json.loads(aux[i]) for i in range(1,len(aux))]
        # params = {'loc':float(aux[1]), 'scale':float(aux[2])}
        params = ast.literal_eval( aux[1] )
        logpriors.append( getattr( st, dist_name )(**params) )
      
    return Xpaths, Xstack, logpriors, maxfiles_bin, logposteriorstack

def predictiveModel( Xstack, N, z, maxfiles ):
    
    # Prediciton
    ies = np.random.choice(np.arange(0,Xstack.shape[0],1), N, replace=False)

    S = np.zeros( [N, z.shape[0], z.shape[1] ] ) # Only lisflood model
    prob = np.zeros( [N, z.shape[0], z.shape[1] ] ) # Lisflood plus inadequacy mean

    for i in range(N):
        
        # Load parameters
        params = Xstack[ ies[i] ]

        # Cond Mean and covariance
        prob[i], S[i] = predBC( z, params, maxfile=maxfiles[ ies[i] ] )

    return prob, S

def indexMap():
    return 0


## =============================================================================

# Read MCMC
# folder = '/home/labdin-i9/Desktop/MBalbi/BuscotProbitR/results/Probit_0511_i9'
folder = 'C:\\Users\\Admin\\Desktop\\MBalbi\\BuscotProbit\\results\\BC_040722_U2'
# folder = 'C:\\Users\\maria\\OneDrive - fi.uba.ar\\PhD\Projects\\Probabilistic flood hazard analysis\\models\\Lisflood\\BuscotProbit\\results\\BC_2701_U2'

Xpaths, Xstack, logpriors, maxfiles, logp = read_mcmc_results( folder, burnin=7500,
                                                               nvars=4 )

# Compute & Plot diagnostics
R, var_j, rhot, neff = paths_diagnostics( Xpaths[7500:,:,:], True, *logpriors )
plt.show()

# Training Data
# bounds = [22,45,0,30]
bounds = [9,45,0,68]
src = gdal.Open( 'Observations//BuscotFlood92.tiff' )
z = src.GetRasterBand(1).ReadAsArray()
src = None
# DEM
dem = 'Buscot.dem.asc'
src = gdal.Open( dem )
dem = src.GetRasterBand(1).ReadAsArray()
src = None

# Predictions
N = 1000
pmap, S = predictiveModel( Xstack, N, z, maxfiles )

# Probability of flooding (P[h>threshold])
S0 = S>0.01
Pmap = S0.mean(axis=0)

# Best prediciton
best_fit_index = np.where(logp==logp.max())[0][0]
Xstack_best = np.array([ Xstack[best_fit_index,:] ])
_, S_best = predictiveModel( Xstack_best, 1, z, maxfiles )
S_best = S_best[0]

# Goodness-of-fit map
gof = Pmap * ( z - 1 ) + ( 1 - Pmap ) * z
gof_total = np.mean( np.abs(gof[bounds[0]:bounds[1],bounds[2]:bounds[3]]) )

# Posterior marginal densities
# var_names = [ r'$\log\ r_{ch}$', r'$\log\ r_{fp}$', 
#           r'$\Phi^{-1} ( \sigma_1 )$', r'$\Phi^{-1} ( \sigma_2 )$' ]
var_names = [ r'$\ r_{ch}$', r'$\ r_{fp}$', 
          r'$ \sigma_1 $', r'$\sigma_2$' ]
fig, ax = plt.subplots( figsize=(8.3/2.54,11.5/2.54), ncols=1, nrows=len(logpriors) )
transform = lambda x: [ np.log(x), np.log(x), st.norm.ppf(x), st.norm.ppf(x) ]
transform_inv = lambda x: [ np.exp(x), np.exp(x), st.norm.cdf(x), st.norm.cdf(x) ]
transform_dv = lambda x: [ 1/x, 1/x, 1/st.norm.pdf( st.norm.ppf(x) ), 1/st.norm.pdf( st.norm.ppf(x) ) ]
x_bounds = [(0.01,0.1), (0.01,0.1), (0.5,1), (0.5,1) ]
for i in range( len(logpriors) ):
    mu, sig = logpriors[i].stats()
    # x = transform_inv( np.arange( mu-1.5*np.sqrt(sig), mu+1.5*np.sqrt(sig), sig/100) )[i]
    x = np.arange(x_bounds[i][0], x_bounds[i][1], 0.001)
    o2 = ax[i].fill_between( x, transform_dv( x )[i]*logpriors[i].pdf( transform(x)[i] ), 
                             alpha=.5, label='prior' )
    ax[i] = sns.kdeplot( transform_inv(Xstack[:,i])[i] , linewidth=0, ax=ax[i], fill=True, alpha=0.5,
                         label='posterior', bw_adjust=4.5)
    ax[i].set_xlabel( var_names[i] )

ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.5), ncol=2)
fig.tight_layout()
fig.subplots_adjust( hspace=0.55 )
# fig.savefig( 'C:\\Users\\Admin\\OneDrive - fi.uba.ar\\PhD\\Projects\\Bayesian calibration of inundation models\\figures_for_paper\\BC_marginal_posteriors_full.pdf', format='pdf', dpi=300)
plt.show()

# Joint posterior of simulator parameters
Z_obs = z[bounds[0]:bounds[1],bounds[2]:bounds[3]]
X0 = np.exp(Xstack[:,:2])
df = pd.DataFrame( Xstack, columns=['rch','rfp','alpha','beta'])

# fig, axes = plt.subplots( figsize=(12/2.54,12/2.54), ncols=4, nrows=4,
#                           sharex='col' )
# for i in range( len(logpriors) ):
#     # Diag
#     mu, sig = logpriors[i].stats()
#     xi = np.arange( mu-2.5*np.sqrt(sig), mu+2.5*np.sqrt(sig), sig/100)
#     axes[i,i].fill_between( xi, logpriors[i].pdf(xi), alpha=.5,
#                               linewidth=0.5 )
#     sns.kdeplot( Xstack[:,i] , linewidth=0, ax=axes[i,i], fill=True,
#                  alpha=0.5 )
#     axes[i,i].set_ylabel('')
#     for j in range( i+1,len(logpriors) ):
#         # Upper
#         sns.kdeplot( x=Xstack[:,j], y=Xstack[:,i], ax=axes[i,j], fill=True,
#                      alpha=0.5 )
#         # Lower
#         sns.kdeplot( x=Xstack[:,i], y=Xstack[:,j], ax=axes[j,i], fill=True,
#                 alpha=0.5 )
#         # s
#         axes[i,j].set_yticklabels([])
#         axes[j,i].set_yticklabels([])
#         axes[j,j].set_yticklabels([])
#     # change labels
#     axes[i,0].set_ylabel( var_names[i] )
#     axes[-1,i].set_xlabel( var_names[i] )

# fig.tight_layout()
# plt.show()

# df = pd.DataFrame( np.exp(Xstack), columns=['rch','rfp','alpha','beta'])
# g = sns.JointGrid( data=df, x='rch', y='rfp', height=8.3/2.54,
#                    xlim=[0,0.15], ylim=[0,0.15], space=0, ratio=3,
#                    marginal_ticks=False )
# g.plot_joint( sns.kdeplot, cmap='viridis' )
# g.plot_marginals( sns.kdeplot, fill=True, lw=0 )
# mu, sig = logpriors[0].stats()
# xi = np.arange( np.exp(mu)-2.5*np.sqrt(sig)*np.exp(mu), np.exp(mu)+4.5*np.sqrt(sig)*np.exp(mu), sig/100)
# g.ax_marg_x.fill_between( xi, st.lognorm.pdf(xi,s=np.sqrt(sig),scale=np.exp(mu)), alpha=.5, linewidth=0 )
# mu, sig = logpriors[1].stats()
# xi = np.arange( np.exp(mu)-2.5*np.sqrt(sig)*np.exp(mu), np.exp(mu)+4.5*np.sqrt(sig)*np.exp(mu), sig/100)
# g.ax_marg_y.fill_between( st.lognorm.pdf(xi,s=np.sqrt(sig),scale=np.exp(mu))[::-1], xi[::-1], alpha=.5, linewidth=0 )
# g.set_axis_labels( xlabel=r'$r_{ch}$', ylabel=r'$r_{fp}$' )
# g.refline( x=np.exp(Xstack_best[0][0]), y=np.exp(Xstack_best[0][1]) )
# g.fig.tight_layout()

# Best prediciton
fig, ax = plt.subplots( figsize=(8.3/2.54,5/2.54), ncols=1, nrows=1 )
data = S_best[bounds[0]:bounds[1],bounds[2]:bounds[3]]
masked_data = np.ma.masked_where( data <= 0, data)
o2 = ax.imshow( dem[bounds[0]:bounds[1],bounds[2]:bounds[3]], cmap=cm.gray,
                   extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05])
ax.set_xlabel( 'x distance [km]' )
ax.set_ylabel( 'y distance [km]' )
cmap = cm.Blues  # Can be any colormap that you want after the cm
cmap.set_bad(color=[0.4,0.4,0.4])
image = ax.imshow( masked_data, cmap=cmap, interpolation='none',
                      extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05])
bar2 = fig.colorbar( image, ax=ax )
bar2.set_label('Flood depth [m]')
ax.contour( Z_obs[::-1,:],
               levels=[.9999, 1.0001], colors='k', linestyles='--',
               extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05])

# Probability of flood
fig, ax = plt.subplots( figsize=(8.3/2.54,7.5/2.54), ncols=1, nrows=2, sharex='col' )
o3 = ax[0].imshow( Pmap[bounds[0]:bounds[1],bounds[2]:bounds[3]],
                   extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05] )
ax[0].contour( Z_obs[::-1,:],
               levels=[.9999, 1.0001], colors='k', linestyles='--',
               extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05])
# ax[0].set_xlabel( 'x distance [km]' )
ax[0].set_ylabel( 'y distance [km]' )
bar3 = fig.colorbar( o3, ax=ax[0] )
bar3.set_label('probability of flood')
# Goodness-of-fit
o4 = ax[1].imshow( gof[bounds[0]:bounds[1],bounds[2]:bounds[3]],
                   extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05], cmap='bwr',
                   vmin=-1, vmax=1)
ax[1].contour( Z_obs[::-1,:], 
               levels=[.9999, 1.0001], colors='k', linestyles='--',
               extent=[0,(bounds[3]-bounds[2])*0.05,0,(bounds[1]-bounds[0])*0.05] )
ax[1].set_xlabel( 'x distance [km]' )
ax[1].set_ylabel( 'y distance [km]' )
bar4 = fig.colorbar( o4, ax=ax[1], ticks=np.arange(-1,1,0.5) )
bar4.set_label(r'$\leftarrow$ overpred   $\rho$   underpred $\rightarrow$')

# Settings
fig.tight_layout()

# fig.savefig( 'C:\\Users\\Admin\\OneDrive - fi.uba.ar\\PhD\\Projects\\Bayesian calibration of inundation models\\figures_for_paper\\BC_results_full.pdf', format='pdf', dpi=300)
plt.show()

# Flood-depth histogram at point
# Ss = S[:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
# col = 15
# row = 15
# fig, ax = plt.subplots( figsize=(8.3/2.54,4/2.54) )
# ax = sns.kdeplot( Ss[:,row,col] , linewidth=0, ax=ax, fill=True, alpha=0.5,
#                   label='posterior')
# plt.show()

