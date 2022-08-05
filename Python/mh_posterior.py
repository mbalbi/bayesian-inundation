"""
Sampling from a posterior density:

- Importance sampling resampling
- Metropolis-Hastings

"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import csv
import os
import scipy.stats.kde as kde
import seaborn as sns



def isr_sampler(Nsim, target_pdf, importance_pdf, resampling=1):
    """
    Importance sampling resampling algorithm for posterior sampling
    
    args:
        - Nsim
        - target_pdf
        - importance_pdf
        - resampling
    output:
        - x
        - r
        - wr
    """
    
    # Sample from the importance function
    r = importance_pdf.rvs(size=Nsim)
    h = importance_pdf.pdf(r)
    # Evaluate the posterior at each sample
    p = np.array([])
    for re in r:
        p = np.append(p,target_pdf(re))
    # Compute weights
    w = p / h
    # Renormalize weights
    wr = w/sum(w)
    # Scaling constant for pdf
    c = sum(w)/Nsim
    # Extract with replacement
    i = np.random.choice(np.arange(Nsim), int(Nsim*resampling), p=wr)
    x = r[i]
    return x, r, wr, c


def mh_sampler(vars, target_logpdf, Nsim, x0, q_cov, burnin=0,
               resultscsv='mh_summary.csv'):
    """
    """
    # Number of parameters
    Nvars = len(x0)
    
    # Flood files names
    lfpname = resultscsv[:-4]
    
    # Create datafra
    with open(resultscsv, "w", newline='') as f:
        # Writers
        writer = csv.writer(f, delimiter=',', quotechar='"')
        # Headers
        writer.writerow( vars + ['r', 'accepted', 'logpdf', 'mapname'] )
        f.flush()
    
        # Run paths
        X = np.zeros([Nsim, Nvars])
        acceptance = np.zeros(Nsim)
        x_prev = x0
        logpdf_prev, _ = target_logpdf( x_prev, output=lfpname+'_sim0.max' )
        
        # Write in csv
        row = [str(a)[:9] for a in x_prev] + \
              ['-', '-', str(logpdf_prev)[:7], lfpname+'_sim0.max' ]
        writer.writerow( row )
        f.flush()

        # Iterations
        for i in range(Nsim):
            # Sample proposal x from jumping distribution
            x_prop = st.multivariate_normal.rvs(mean=x_prev, cov=q_cov, size=1)
            
            # Calculate ratio of the densities
            logpdf_prop, _ = target_logpdf( x_prop,
                                        output=lfpname+'_sim'+str(i+1)+'.max' )
            r = np.exp( logpdf_prop + \
                st.multivariate_normal.logpdf( x_prop, mean=x_prev, cov=q_cov ) - \
                logpdf_prev - \
                st.multivariate_normal.logpdf( x_prev, mean=x_prop, cov=q_cov ) )
            
            # Accept with probability r
            if np.random.rand() <= min(r,1):
                x_new = x_prop
                logpdf_prev = logpdf_prop
                acceptance[i] = 1
            else:
                x_new = x_prev
                acceptance[i] = 0
                
            # Update for next it
            X[i,:] = x_new
            x_prev = x_new
            
            # Write in csv
            row = [str(a)[:9] for a in x_prop] + \
                  [str(r)[:9], str(acceptance[i]), str(logpdf_prop)[:5],
                   lfpname+'_sim'+str(i+1)+'.max']
            writer.writerow( row )
            f.flush()

    # Remove burn-in samples
    Xbin = X[burnin:,:]
    
    # Acceptance rate
    acc_rate = acceptance.cumsum(0) / (np.arange(acceptance.shape[0]) + 1)
    acc_rate = acc_rate[burnin:]
    
    return X, Xbin, acc_rate

def adaptive_mh_sampler(vars, target_logpdf, Nsim, x0, q_cov,
                        tune=1000, tune_intvl=30, burnin=0, 
                        resultscsv='mh_summary.csv'):
    """
    
    args:
        - target_logpdf:
        - Nsim: Number of simulation (after tuning)
        - x0: Initial point
        - S0: Initial covariance matrix
        - tune: Number of samples for tuning phase
        - tune_intvl: Number of samples between updating of covariance matrix
        - burnin: Burn-in sample length (after tuning)
    
    out:
        - X
        - Xbin:
        - 
    
    """
    # Number of parameters
    Nvars = len(x0)
    
    # Initial covariance matrix
    S = q_cov
    
    # Initial scaling factor
    gamma = 1
    
    # Run intervals of tuning period
    for i in range( tune//tune_intvl ):
        
        X, Xbin, acc_rate = mh_sampler(vars, target_logpdf, tune_intvl, x0,
                                       gamma*S, burnin=0,
                                       resultscsv=resultscsv)
        
        # Update jumping covariance
        # X = np.unique( X, axis=0 ) # Only accepted samples
        S_new = np.cov( X.T )
        if np.linalg.det(S_new) < 1e-10:
            S = S
        else:
            S = S_new
        
        # Update new point
        x0 = X[-1]
            
        if acc_rate[-1] < 0.001:
            # reduce by 90 percent
            gamma *= 0.1
        elif acc_rate[-1] < 0.05:
            # reduce by 50 percent
            gamma *=  0.5
        elif acc_rate[-1] < 0.2:
            # reduce by ten percent
            gamma *=  0.9
        elif acc_rate[-1] > 0.95:
            # increase by factor of ten
            gamma *=  10.0
        elif acc_rate[-1] > 0.75:
            # increase by double
            gamma *=  2.0
        elif acc_rate[-1] > 0.5:
            # increase by ten percent
            gamma *=  1.1
    
    # Simulation phase
    X, Xbin, acc_rate = mh_sampler(vars, target_logpdf, Nsim, x0, gamma*S,
                                   burnin=burnin, resultscsv=resultscsv)
    
    return X, Xbin, acc_rate
    
def mh_paths_sampler(vars, target_logpdf, Npaths, Nsim, x0, q_cov,
                     burnin=0, thin=1, sampler='normal',
                     tune=1000, tune_intvl=30, outputDir=''):
    """
    Markov-Chain Monte-Carlo sampler using Metropolis-Hastings algorithm for posterior sampling
    
    args:
        - target_logpdf:
        - jump_rule:
        - Npaths:
        - Nsim: Number of simulations per path
        
    out:
        - X
        - Xbin
        - Xpaths
        - acceptance
        
    """    
    # Number of parameters
    Nvars = len(x0[0])
    
    # Check that Npaths and x0 are the same dimension
    if not len(x0) > Npaths:
        return -1, -1, -1, -1
    
    # Run paths
    X = np.zeros([Nsim, Nvars, Npaths])
    Xbin = np.zeros([Nsim-burnin, Nvars, Npaths])
    acc_rate = np.zeros([Nsim-burnin, Npaths])
    for i in range(Npaths):
        print('Starting path {} of {}'.format(str(i+1), str(Npaths)))
        # Results for current path
        filename = os.path.join(outputDir, 'Path'+str(i)+'_results.csv')
        # Sampler type
        if sampler == 'normal':
            X[:,:,i], Xbin[:,:,i], acc_rate[:,i] = mh_sampler(
                                                        vars,
                                                        target_logpdf,
                                                        Nsim, x0[i],
                                                        q_cov, burnin,
                                                        resultscsv=filename
                                                        )
        elif sampler == 'adaptive':
            X[:,:,i], Xbin[:,:,i], acc_rate[:,i] = adaptive_mh_sampler(
                                                        vars,
                                                        target_logpdf, Nsim,
                                                        x0[i], q_cov, tune=tune,
                                                        tune_intvl=tune_intvl, 
                                                        burnin=burnin,
                                                        resultscsv=filename
                                                        )
    
    # Remove trivial paths (that never vary)
    delete_indices = []
    for i in range(Npaths):
        path = Xbin[:,0,i]
        if np.all( path == path[0] ):
            delete_indices.append( i )
    Xbin = np.delete( Xbin, delete_indices, axis=2 )

    # Compute mixing and convergence properties of the chains
    R, var_j, r_hot, n_eff = paths_diagnostics(Xbin, False)
    
    # Thinning of paths
    Xbin = Xbin[::thin]
    
    # Stack paths
    Xstack = Xbin[:,:,0]
    for i in range( 1, Xbin.shape[2] ):
        xpath = Xbin[:,:,i]
        Xstack = np.vstack( [Xstack, xpath] )
    
    return Xbin, Xstack, acc_rate, (R, var_j, r_hot, n_eff)
    
def predictive_posterior_sampler(distr, Xbin):
    """
    """
    sample = np.array([])
    for param in Xbin:
        loc, scale, *arg = param   
        sample = np.append( sample, distr.rvs(loc, scale, *arg, size=1) )
    return sample

def paths_diagnostics( Xpaths, plot, *arg ):
    
    # Compute mixing and convergence for each variable
    n = Xpaths.shape[0]
    m = Xpaths.shape[2]
    Nvars = Xpaths.shape[1]

    #  Within-paths mean
    phat_j = np.mean( Xpaths, axis=0 )

    # between-paths variance
    B = n * np.var( phat_j, axis=1, ddof=1 )

    # Within-paths variance
    sj2 = np.var( Xpaths, axis=0, ddof=1 )

    # Within-paths variance
    W = np.mean( sj2, axis=1 )

    # Marginal posterior variance of estimand
    var_j = (n-1)/n * W + 1/n * B

    # Potential scale reduction (Gelman-Rubin coefficient)
    R = np.sqrt( var_j/W )
    
    # Variogram and correlation
    max_lag = n
    Vt = np.zeros([n, Nvars])
    for t in range(0,n):
        Vti = np.mean( np.mean( (Xpaths[:(n-t)] - Xpaths[t:])**2, 0 ), 1)
        Vt[t] = Vti

    # Autocorrelation
    rhot = 1 - Vt/2/var_j

    # Effective number of samples
    aux = rhot[1:] + rhot[:-1]
    T = np.zeros( Nvars )
    neff = np.zeros( Nvars )
    for i in range(Nvars):
        indices = np.where( aux[:,i]<=0 )[0]
        if indices.any():
            T[i] = indices[0]
        else:
            T[i] = -1
            neff[i] = 0
            continue
        # Effective number of samples 
        neff[i] = n*m/(1+2*rhot[:int(T[i]),i].sum())

    if plot:
        plot_diagnostics( Xpaths, R, *arg)

    return R, var_j, rhot, neff

def plot_diagnostics( Xpaths, R, *args):
    
    # Logpriors
    logPriors = args
    
    # Stack paths
    Xstack = Xpaths[:,:,0]
    for i in range( 1, Xpaths.shape[2] ):
        xpath = Xpaths[:,:,i]
        Xstack = np.vstack( [Xstack, xpath] )
    
    # Create figure
    fig, ax = plt.subplots(nrows=Xpaths.shape[1], ncols=3, figsize=(14,9))
    fig.tight_layout()
    fig.subplots_adjust( wspace=0.1, hspace=0.15)

    # Plot
    for i in range(Xpaths.shape[1]):
        for j in range(Xpaths.shape[2]):
            ax[i,0] = sns.kdeplot( Xpaths[:,i,j] , linewidth=0.5, ax=ax[i,0] )
        # Histogram
        o_post = ax[i,0].hist( Xstack[:,i], density='True', bins=40, alpha=0.4 )
        o_prior = ax[i,0].plot( o_post[1], logPriors[i].pdf(o_post[1]) )
        # Sampling history
        o_mix = ax[i,1].plot( Xpaths[:,i,:], linewidth=0.5 )
        ax[i,1].text(.5,.9, 'R={}'.format(str(R[i])[:4]),
                    horizontalalignment='center', transform=ax[i,1].transAxes)
        # Sampling autocorrelation
        o_acorr = ax[i,2].acorr( Xpaths[:,i,0], usevlines=True, normed=True,
                                    maxlags=50, lw=2 )
        
    return

def hpd_grid(sample, alpha=0.05, roundto=2):
    """
    
    Calculate highest posterior density (HPD) of array for given alpha. 
    The HPD is the minimum width Bayesian credible interval (BCI). 
    The function works for multimodal distributions, returning more than one mode
    Parameters
    ----------
    sample : Numpy array or python list
        An array containing MCMC samples
    alpha : float
        Desired probability of type I error (defaults to 0.05)
    roundto: integer
        Number of digits after the decimal point for the results
    Returns
    ----------
    hpd: array with the lower 
          
    """
    sample = np.asarray(sample)
    sample = sample[~np.isnan(sample)]
    # get upper and lower bounds
    l = np.min(sample)
    u = np.max(sample)
    density = kde.gaussian_kde(sample)
    x = np.linspace(l, u, 2000)
    y = density.evaluate(x)
    #y = density.evaluate(x, l, u) waitting for PR to be accepted
    xy_zipped = zip(x, y/np.sum(y))
    xy = sorted(xy_zipped, key=lambda x: x[1], reverse=True)
    xy_cum_sum = 0
    hdv = []
    for val in xy:
        xy_cum_sum += val[1]
        hdv.append(val[0])
        if xy_cum_sum >= (1-alpha):
            break
    hdv.sort()
    diff = (u-l)/20  # differences of 5%
    hpd = []
    hpd.append(round(min(hdv), roundto))
    for i in range(1, len(hdv)):
        if hdv[i]-hdv[i-1] >= diff:
            hpd.append(round(hdv[i-1], roundto))
            hpd.append(round(hdv[i], roundto))
    hpd.append(round(max(hdv), roundto))
    ite = iter(hpd)
    hpd = list(zip(ite, ite))
    modes = []
    for value in hpd:
         x_hpd = x[(x > value[0]) & (x < value[1])]
         y_hpd = y[(x > value[0]) & (x < value[1])]
         modes.append(round(x_hpd[np.argmax(y_hpd)], roundto))
    return hpd, x, y, modes

## =============================================================================
if __name__ == '__main__':
    # Target density
    m = np.zeros(2)
    rho = 0.9
    cov = np.array([[1, rho],[rho, 1]])
    def target_pdf(x): # Unscaled Posterior
        y = [15420, 19787, 25050, 22413, 20430, 25500, 14277, 21027, 12967,
        20427, 16007, 20700, 20000, 21127, 18603, 13897, 22020, 18447,
        20280, 18447, 20280, 16200, 16440, 19647, 21880, 16067, 13787,
        19673, 16600, 16560, 17853, 25510]
        lik = [st.dweibull.pdf(yi, loc=x[0], scale=x[1], *(x[2],)) for yi in y]
        pdf =  np.prod(lik)
        return pdf
    # Jumping distribution
    def jump_pdf(x, x_prev):
        pdf =  st.multivariate_normal.pdf(x, mean=x_prev,
                                cov=np.array([1000**2,100**2,0.5])*np.eye(3))
        return pdf
    def jump_sampler(x, size):
        r =  st.multivariate_normal.rvs(mean=x,
                                cov=np.array([1000**2,100**2,0.5])*np.eye(3), 
                                size=size)
        return r    

    Nsim = 40
    burnin = 39
    # Starting point
    Npaths = 1000
    x0 = st.multivariate_normal.rvs(mean=np.array([18937,3141,1.566]), 
                                cov=np.array([1000**2,100**2,0.5])*np.eye(3), 
                                size=Npaths+1)
    x, xbin, acc = mh_sampler(Npaths, burnin, x0, Nsim, target_pdf, jump_pdf, 
                              jump_sampler)

    # Compute quantile for each sample from the posterior (1.5 yrs)
    pT = 1 - 1/1.5
    Qbankfull = []
    for i in range(np.shape(xbin)[0]):
        xi = xbin[i,:]
        Qbankfull.append(st.dweibull.ppf(pT, loc=xi[0], scale=xi[1], *(xi[2],)))

    plt.hist(Qbankfull, density=True, bins=5, histtype='stepfilled', alpha=0.2)
    plt.show()

    # PLOT
    # Plot histogram
    # plt.hist2d(x[:,0], x[:,1], bins=20, density=True)
    # xplot = np.linspace(-3,3,300)
    # yplot = np.linspace(-3,3,300)
    # X, Y = np.meshgrid(xplot, yplot)
    # Z = 0 * X
    # for i in range(np.shape(X)[0]):
    #     for j in range(np.shape(X)[1]):
    #         Z[i,j] = target_pdf([X[i,j],Y[i,j]])
    # plt.contour(X, Y, Z)

    # plt.hist(x, density=True, bins=50, histtype='stepfilled', alpha=0.2)
    # xplot = np.linspace(-5,5,500)
    # plt.plot(xplot, target_pdf(xplot)/2.5)
    # Plot samples

    # Plot random walk
    # fig, ax = plt.subplots(1,2, sharey=True, sharex=True)
    # ax[0].plot(x[:,0], x[:,1], marker='o', linewidth=0)
    # x_anal = st.multivariate_normal.rvs(mean=m, cov=cov, size=int((Nsim-warmup)*paths))
    # ax[1].plot(x_anal[:,0],x_anal[:,1], marker='o', linewidth=0)
    # plt.show()
