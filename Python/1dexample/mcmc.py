import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.stats.kde as kde
import seaborn as sns

def mcmc_sampler( Nsim, jump_cov, posterior, xprev ):
    '''
    Metropolis-Hastings algorithm, with Gaussian jump distribution.
    
    Args:
        - Nsim: Number of simulations per path
        - jump_cov: covariance matrix for gaussian jump distribution (n x n)
        - logpriors
        
    Returns:
        -
    '''
    Nvars = xprev.shape[0]
    
    # Initializ output array
    X = np.zeros([Nsim, Nvars])
    validation = np.zeros( [Nsim,3] )
    accepted = 0
        
    # Initial sample
    logpdf_prev = posterior.logpdf( xprev[0], xprev[1:] )
    # Iterations
    for i in range(Nsim):
        
        # Sample proposal x from jumping distribution
        xprop = st.multivariate_normal.rvs( mean=xprev, cov=jump_cov, size=1 )
        
        # Calculate ratio of the densities
        logpdf_prop = posterior.logpdf( xprop[0], xprop[1:] )
        r = np.exp( logpdf_prop + \
            st.multivariate_normal.logpdf( xprop, mean=xprev, cov=jump_cov ) - \
            logpdf_prev - \
            st.multivariate_normal.logpdf( xprev, mean=xprop, cov=jump_cov ) )
        
        # Accept with probability r
        u = np.random.rand()
        if u <= min(r,1):
            x_new = xprop
            logpdf_prev = logpdf_prop
            accepted += 1
        else:
            x_new = xprev
            
        # Update for next it
        X[i,:] = x_new
        validation[i,0] = logpdf_prev
        validation[i,1] = r
        validation[i,2] = u
        xprev = x_new
    
    return X, accepted/(Nsim+1), validation

def adaptive_mcmc_sampler( Nsim, jump_cov, posterior, xprev,
                           tune=1000, tune_intvl=30 ):
    """
    
    args:
        - Nsim:
    
    out:
        - X:
    
    """  
    # Initial covariance matrix
    S = jump_cov
    
    # Initial scaling factor
    gamma = 1
    x0 = xprev
    
    # Run intervals of tuning period
    for i in range( tune//tune_intvl ):
        
        X, acc_rate, _ = mcmc_sampler( tune_intvl, S, posterior, xprev=x0 )
        
        # Update jumping covariance
        # X = np.unique( X, axis=0 ) # Only accepted samples
        S_new = np.cov( X.T )
        if np.linalg.det(S_new) < 1e-10:
            S = S
        else:
            S = S_new
        
        # Update new point
        x0 = X[-1]
            
        if acc_rate < 0.001:
            # reduce by 90 percent
            gamma *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            gamma *=  0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            gamma *=  0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            gamma *=  10.0
        elif acc_rate > 0.75:
            # increase by double
            gamma *=  2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            gamma *=  1.1
    
    # Simulation phase
    X, acc_rate, logpdf = mcmc_sampler( Nsim, gamma*S, posterior, xprev=X[-1] )
    
    return X, acc_rate, logpdf


def paths_sampler( Nsim, Npaths, posterior, x0, jump_cov, burnin, thin=1,
                   type='normal', **kwargs ):
    
    X = np.zeros([Nsim, jump_cov.shape[0],Npaths])
    valid = np.zeros([Nsim,3,Npaths])
    for j in range(Npaths):
        if type == 'normal':
            X[:,:,j], _, valid[:,:,j] = mcmc_sampler( Nsim, jump_cov, posterior,
                                                      x0[j] )
        elif type == 'adaptive':
            X[:,:,j], _, valid[:,:,j] = adaptive_mcmc_sampler( Nsim, jump_cov, 
                                                               posterior,
                                                               x0[j],
                                                               **kwargs )
    # Stack
    Xp = X[burnin::thin,:,:]
    Xstack = np.vstack( [Xp[:,:,0],Xp[:,:,1]] )
    valid_stack = np.vstack( [valid[burnin::thin,:,0], valid[burnin::thin,:,1]] )
    
    return Xstack, valid_stack, Xp

def paths_diagnostics( Xpaths, plot, prior ):
    
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
        plot_diagnostics( Xpaths, R, prior)

    return R, var_j, rhot, neff

def plot_diagnostics( Xpaths, R, prior):
    
    # Stack paths
    Xstack = Xpaths[:,:,0]
    for i in range( 1, Xpaths.shape[2] ):
        xpath = Xpaths[:,:,i]
        Xstack = np.vstack( [Xstack, xpath] )
    
    # Create figure
    fig, ax = plt.subplots(nrows=Xpaths.shape[1], ncols=2, figsize=(14,9))
    fig.tight_layout()
    fig.subplots_adjust( wspace=0.1, hspace=0.15)

    # Plot
    for i in range(Xpaths.shape[1]):
        for j in range(Xpaths.shape[2]):
            aux = sns.kdeplot(  Xpaths[:,i,j], linewidth=0.5, ax=ax[i,0] )
        # Histogram
        o_post = ax[i,0].hist( Xstack[:,i], density='True', bins=40, alpha=0.4 )
        o_prior = ax[i,0].plot( o_post[1], prior.distributions[i].pdf( o_post[1]) )
        # Sampling history
        o_mix = ax[i,1].plot( Xpaths[:,i,:], linewidth=0.5 )
        ax[i,1].text(.5,.9, 'R={}'.format(str(R[i])[:4]),
                    horizontalalignment='center', transform=ax[i,1].transAxes)
        # Sampling autocorrelation
        # o_acorr = ax[i,2].acorr( Xpaths[:,i,0], usevlines=True, normed=True,
        #                             maxlags=50, lw=2 )
        
    return
