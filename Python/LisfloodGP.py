
"""

Model

z = M(x,t) + d(t) + e

where d(t) and its prior can be described as Gaussian Processes:
d ~ N(hd.T * betad, Vd)
mean[d] = sum(hd,betad)
cov[d] = Vd

where hd are the 'basis functions' for the model inadequacy and Vd is the
covariance function described by hyperparameters xid

Likelihood of observed data:

# Covariance model for inadequacy

Md = betax*x + betay*y
Vd = sd^2 * exp(-wx*|x-x'|^2 - wy*|y-y'|^2)

List of parameters to calibrate:
- Flood model:
    r_ch: channel roughness
    r_fp: floodplain roughness
- Inadequacy function:
    betax, betay: Inadequacy mean parameters
    wx, wy: length inverse
    sd: standard deviation
- Error model:
    sigma: Noise variance

"""

import numpy as np
import scipy.stats as st
# from qsimvnauto import qsimvnauto

from preprocess import create_par, wait_timeout

from minimax_tilting_sampler import TruncatedMVN

import subprocess, os
from shutil import copy

import pickle

import osgeo.gdal as gdal
import osgeo.osr as osr

import time

def mask_raster(base_raster, mask_raster, band1=1, band2=1, reverse=False):
    # Base raster
    base_transform = base_raster.GetGeoTransform()
    base_array = base_raster.GetRasterBand(band1).ReadAsArray()
    # Xcount = base_raster.RasterXSize # number of cols
    # Ycount = base_raster.RasterYSize # number of rows

    # Mask raster
    mask_transform = mask_raster.GetGeoTransform()
    mask_array = mask_raster.GetRasterBand(band2).ReadAsArray()

    return mask_arrays(base_array, mask_array, reverse=reverse)

def mask_arrays(base_array, mask_array, reverse=False):
    Xcount = np.size(base_array,1)
    Ycount = np.size(base_array,0)
    # Loop through base array
    new_array = np.zeros([Ycount,Xcount])
    count = 0
    for i in range(Ycount):
        for j in range(Xcount):
            chanmask_value = mask_array[i, j]
            if reverse:
                if chanmask_value > 0:
                    count += 1
                    new_array[i,j] = base_array[i,j]
            elif not reverse:
                if chanmask_value <= 0:
                    new_array[i,j] = base_array[i,j]

    return(new_array)

def NoiseCOV( se, n ):
    return se * np.eye( n )

def GaussianCOV( sd, wx, wy, x_diffs, y_diffs ):
    return sd * np.exp( -wx*(x_diffs)**2 - wy*(y_diffs)**2 )

def mvnormal_logpdf(y, mean, cov):
    dim = len(mean)
    cov_norm = np.linalg.det(cov)
    if cov_norm == 0:
        return -np.inf
    y_norm = y-mean
    aux = np.linalg.solve(cov,y_norm)
    return -1/2*np.log((2*np.pi)**dim * cov_norm) - 1/2 * np.dot(y_norm.T, aux)

def inadequacyMean( x, beta ):
    return 0*x.dot( beta )

def readLisfloodMax( filename, x ):
    src = gdal.Open( filename )
    array = src.GetRasterBand(1).ReadAsArray()
    z = np.array( [ array[x[i][0], x[i][1]] for i in range(len(x))] )
    return z, array

def Lisflood( x, theta, output=None ):
    """
    Returns the flood height for the set of points in x from a Lisflood run
    with parameters theta
    """
    
    # Input raster files
    files = {}
    files['DEMfile'] = 'Buscot.dem.asc'
    files['bcifile'] = 'Buscot.bci' 
    # files['bdyfile'] = 'Buscot.bdy'
    files['SGCwidth'] = 'Buscot.width.asc'
    files['SGCbed'] = 'Buscot.bed.asc'
    files['SGCbank'] = files['DEMfile']
    files['startfile'] = 'Buscot.depth.asc'
    files['weirfile'] = 'Buscot.weir'

    # Settings
    options = {}
    options['resroot'] = 'temp'
    options['dirroot'] = 'temp'
    options['sim_time'] = 1382400
    options['initial_step'] = 1
    options['massint'] = 10*1
    options['saveint'] = options['sim_time']/9
    options['SGCn'] = str(theta[0])
    options['fpfric'] = str(theta[1])
    options['settings'] = ['mint_hk','elevoff']
    
    # Modify BCI file for different input discharges
    # create_bci( bcifile, q )
    
    # Create .par file
    parfile = 'Buscot.par'
    create_par(parfile, files, **options)

    # Run model
    opts = []
    os.environ["OMP_NUM_THREADS"] = str(4)
    # executable = './lisflood_linux' # for Linux
    executable = 'lisflood_intelRelease_double' # for Windows
    call = executable + ' ' + parfile
    proc = subprocess.Popen(call, shell=True)
    result = wait_timeout(proc, 120)

    # Save outputs
    source = os.path.join(options['dirroot'], options['resroot']+'.max')
    if output:
        copy(source, output)
    
    # Extract desired flood heights from .max file
    if x=='array':
        src = gdal.Open( source )
        return src.GetRasterBand(1).ReadAsArray()
    elif x:
        return readLisfloodMax( source, x.astype(int) )

def logPrior( logPriors, params ):
    lps = [ logPriors[i].logpdf(params[i]) for i in range(len(logPriors)) ]
    return sum( lps )

def logLikelihoodM0( x, z, params, output='' ):
    """
    
        - z: Observation at x points
        - x: grid points where values are observed (rows, cols)
        - params: Model parameters (flood model + Inadequacy + Error)
    
    """
    
    # Run LISFLOOD
    S = Lisflood( 'array', np.exp(params[:2]), output=output )
    Sf = S.flatten() # row major style
    
    # Inadequacy function
    d = params[2]*x[:,0] + params[3]*x[:,0] + params[4]*x[:,0]*x[:,1]
    
    # Probit
    logsf = st.norm.logsf( (S.flatten() + d)/np.exp(params[-1]) , loc=0, scale=1 )
    logp = st.norm.logcdf( (S.flatten() + d)/np.exp(params[-1]) , loc=0, scale=1 )
    
    # Log-Likelihood
    logL = np.sum( z.flatten()*logp + (1-z.flatten())*logsf ) 

    return logL, [0,0]

def logLikelihoodBC( x, z, params, output='' ):
    """
    Likelihood for the Binary Channel model:
    P(Z=1|Y=1) = alpha ; P(Z=0|Y=0)=beta
     
        - z: Observation at x points
        - x: grid points where values are observed (rows, cols)
        - params: lisflood_params=params[:2], alpha=params[2], beta=params[3]

    
    """
    
    # Run LISFLOOD
    S = Lisflood( 'array', np.exp(params[:2]), output=output ) 
    
    # Confusion matrix values
    A = np.ma.masked_array( S>0, z==0 ).sum() # Correctly predicted flooded (S>0, z=1)
    B = np.ma.masked_array( S>0, z==1 ).sum() # Overpredicted (S>0, z=0)
    C = np.ma.masked_array( S==0, z==0 ).sum() # Underpredicted (S=0, z=1)
    D = np.ma.masked_array( S==0, z==1 ).sum() # Correctly predicted dry (S=0, z=0)
    
    # Likelihood
    alpha = st.norm.cdf( params[2], loc=0, scale=1 )
    beta = st.norm.cdf( params[3], loc=0, scale=1 )
    # L = alpha**A * (1-alpha)**B * beta**C * (1-beta)**D
    logL = A*np.log(alpha) + B*np.log(1-alpha) + C*np.log(1-beta) + D*np.log(beta)

    return logL, [0,0]


def logLikelihoodProbit( x, z, params, output='' ):
    """
    
        - z: Observation at x points
        - x: grid points where values are observed (rows, cols)
        - params: Model parameters (flood model + Inadequacy + Error)
    
    """
    # Covariance structure
    x_diffs = np.subtract.outer( x[:,1], x[:,1] )
    y_diffs = np.subtract.outer( x[:,0], x[:,0] )
    cov = GaussianCOV( np.exp(params[2]), np.exp(params[3]),
                       np.exp(params[3]), x_diffs, y_diffs )
    cov += NoiseCOV( np.exp(params[4]), len(x) )
    
    # Run LISFLOOD
    S = Lisflood( 'array', np.exp(params[:2]), output=output ) 
    # S = S[10:44,:64]
    S = S[25:40,:25]
    
    # Probit
    inf = np.zeros( z.shape )
    sup = np.zeros( z.shape )
    for i in range( z.shape[0] ):
        if z[i] == 0:
            inf[i] = -np.inf
            sup[i] = 0
        elif z[i] == 1:
            inf[i] = 0
            sup[i] = np.inf
    print('Computing likelihood......')
    t0 = time.time()
    p, status = st.mvn.mvnun( inf, sup, S.flatten(), cov )
    dt = time.time() - t0
    print('Computing time: {}s'.format(str(dt)[:9]))

    return np.log(p), [0,0]

def logLikelihood( x, z, params, output='' ):
    """
    
        - z: Observation at x points
        - x: grid points where values are observed (rows, cols)
        - params: Model parameters (flood model + Inadequacy + Error)
    
    """
    # Covariance structure
    x_diffs = np.subtract.outer( x[:,1], x[:,1] )
    y_diffs = np.subtract.outer( x[:,0], x[:,0] )
    cov = GaussianCOV( np.exp(params[4]), np.exp(params[2]),
                       np.exp(params[3]), x_diffs, y_diffs )
    cov += NoiseCOV( np.exp(params[5]), len(x) )
    
    # Run LISFLOOD
    S = Lisflood( x, np.exp(params[:2]), output=output ) 
    
    # Beta optimal
    Winv = x.T.dot( np.linalg.solve( cov, x ) )
    beta_hat = np.linalg.solve( Winv, x.T.dot( np.linalg.solve( cov, z - 0 ) ) )
    
    # Mean
    mean = S + inadequacyMean( x, beta_hat )
    
    # Log pdf
    logpdf = mvnormal_logpdf( z, mean = mean, cov = cov )
    logpdf += -1/2*np.log( np.linalg.det(Winv) )
    
    return logpdf, beta_hat

def jaccard_fit( z, bounds, params, output='' ):
    """
    Jaccard fit
    
    """
    
    # Run LISFLOOD
    S = Lisflood( "array", params, output=output ) 
    S = S[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    
    # Confusion matrix values
    A = np.ma.masked_array( S>0, z==0 ).sum() # Correctly predicted flooded (S>0, z=1)
    B = np.ma.masked_array( S>0, z==1 ).sum() # Overpredicted (S>0, z=0)
    C = np.ma.masked_array( S==0, z==0 ).sum() # Underpredicted (S=0, z=1)
    D = np.ma.masked_array( S==0, z==1 ).sum() # Correctly predicted dry (S=0, z=0)
    F = (A-B)/(A+B+C)
    
    return {'A':A, 'B':B, 'C':C, 'F':F}

def logPosterior( x, z, logPriors, logLikelihood, params, output, *args ):
    lprior = logPrior( logPriors, params)
    if lprior == -np.inf:
        return -np.inf, (0,0)
    L, beta_hat = logLikelihood( x, z, params, output, *args )
    return L + lprior, beta_hat

def predBC( z, params, maxfile='', prob=True, tol=0.01 ):
    
    # Predictive mean
    src = gdal.Open( maxfile )
    S = src.GetRasterBand(1).ReadAsArray()
    src = None
    
    #
    alpha = st.norm.cdf( params[2], loc=0, scale=1 )
    beta = st.norm.cdf( params[3], loc=0, scale=1 )
    
    if prob:
        pflood = (S > tol)*alpha + (S <= tol)*(beta) # !!!!
    
    return pflood, S

def predProbit( xtest, xtrain, ytrain, params, maxfile='', output=None, bounds=None ):
    """
    Computes the mean and covariance matrix of the regression function f
    at xtest points

    Args:
        xtest: Array of m points (m x d).
        xtrain: Array of n points (n x d)
        ytrain: Array of n points (n x 1)
        theta: Kernel parameters
        sn: Noise level

    Returns:
        (m x n) matrix
    """
    
    # Create covariance matrices
    xtrain_diffs = np.subtract.outer( xtrain[:,1], xtrain[:,1] )
    ytrain_diffs = np.subtract.outer( xtrain[:,0], xtrain[:,0] )
    xtest_diffs = np.subtract.outer( xtest[:,1], xtest[:,1] )
    ytest_diffs = np.subtract.outer( xtest[:,0], xtest[:,0] )
    xtraintest_diffs = np.subtract.outer( xtrain[:,1], xtest[:,1] )
    ytraintest_diffs = np.subtract.outer( xtrain[:,0], xtest[:,0] )
    
    K = GaussianCOV( params[2], params[3], params[3], 
                     xtrain_diffs, ytrain_diffs )
    K += NoiseCOV( params[4], len(xtrain) )
    Kdd = GaussianCOV( params[2], params[3], params[3],
                       xtest_diffs, ytest_diffs )
    Kd = GaussianCOV( params[2], params[3], params[3],
                      xtraintest_diffs, ytraintest_diffs )
    
    # Computational model
    S = Lisflood( "array", params[:2], output=output )
    S = S[bounds[0]:bounds[1],bounds[2]:bounds[3]]
    Sd = S
    # S = readLisfloodMax( maxfile, xtrain.astype(int) ) # already ran
    # S_test = Lisflood( xtest, np.exp(params[0:2]), output=output ) # To run with diff. event
    # Sd = readLisfloodMax( maxfile, xtest.astype(int) )
    
    # Sample from truncated Normal distribution
    yaux = ytrain.flatten()
    inf = np.zeros( yaux.shape[0] )
    sup = np.zeros( yaux.shape[0] )
    for i in range( yaux.shape[0] ):
        if yaux[i] == 0:
            inf[i] = -np.inf
            sup[i] = 0
        elif yaux[i] == 1:
            inf[i] = 0
            sup[i] = np.inf
    tmvn = TruncatedMVN( S.flatten(), 1*K, inf, sup )
    u = tmvn.sample( 1 )
    
    # Inadequacy sampling
    fmean = Kd.dot( np.linalg.solve( K, u - S.flatten() ) )
    fcov = Kdd - Kd.dot( np.linalg.solve( K, Kd.T ) )
    
    return Sd + fmean, fmean, fcov, Sd

def predProbitM0( x, z, params, maxfile='' ):
    
    # Read lisflood output for current params
    src = gdal.Open( maxfile )
    S = src.GetRasterBand(1).ReadAsArray()
    
    # Sample noisy field u
    # d = params[2]*x[:,0] + params[3]*x[:,1] + params[4]*x[:,0]*x[:,1]
    d = params[2]*x[1] + params[3]*x[0] + params[4]*x[0]*x[1]
    
    m = S + d
    se = np.exp(params[-1])
    a = -1000*(z==0) + (0-m)/se*(z>0) # Lower bound
    b = (0-m)/se*(z==0) + 1000*(z>0) # Lower bound
    u = st.truncnorm.rvs( a.flatten(), b.flatten(), loc=m.flatten(),
                          scale=se*np.ones(np.size(S)),
                          size=np.size(S) )
    u = u.reshape( z.shape[0], z.shape[1] )
    
    # Sample inadequacy field from noisy field
    # f = st.norm.rvs( loc=u-S-d, scale=1, size=1 )
    
    # Sample flood field from u
    f = np.maximum(m,0)
    
    return f, S, d, u

def predGP( xtrain, xtest, ztrain, params, maxfile='', output=None,
            log_mode=False):
    
    # Create covariance matrices
    xtrain_diffs = np.subtract.outer( xtrain[:,1], xtrain[:,1] )
    ytrain_diffs = np.subtract.outer( xtrain[:,0], xtrain[:,0] )
    xtest_diffs = np.subtract.outer( xtest[:,1], xtest[:,1] )
    ytest_diffs = np.subtract.outer( xtest[:,0], xtest[:,0] )
    xtraintest_diffs = np.subtract.outer( xtrain[:,1], xtest[:,1] )
    ytraintest_diffs = np.subtract.outer( xtrain[:,0], xtest[:,0] )
    
    cov_train = GaussianCOV( np.exp(params[4]), np.exp(params[2]),
                             np.exp(params[3]), xtrain_diffs, ytrain_diffs )
    cov_train += NoiseCOV( np.exp(params[5]), len(xtrain) )
    cov_test = GaussianCOV( np.exp(params[4]), np.exp(params[2]),
                             np.exp(params[3]), xtest_diffs, ytest_diffs )
    cov_traintest = GaussianCOV( np.exp(params[4]), np.exp(params[2]),
                                 np.exp(params[3]), 
                                 xtraintest_diffs, ytraintest_diffs )

    # Predictive mean
    S_train = readLisfloodMax( maxfile, xtrain.astype(int) ) # already ran
    # S_test = Lisflood( xtest, np.exp(params[0:2]), output=output ) # To run with diff. event
    S_test = readLisfloodMax( maxfile, xtest.astype(int) )
    if log_mode:
        S_train = np.log( S_train )
        S_train[ S_train == -np.inf ] = -10
        S_test = np.log( S_test )
        S_test[ S_test == -np.inf ] = -10
    m_train = S_train
    m_train += inadequacyMean( xtrain, params[6:8] )
    m_test = S_test 
    m_test += inadequacyMean( xtest, params[6:8] )
    L = np.linalg.cholesky( cov_train )
    alpha = np.linalg.solve( L.T, np.linalg.solve( L, ztrain-m_train ) )
    f = cov_traintest.T.dot( alpha ) + m_test
    # f = m_test

    # Predictive variance
    v = np.linalg.solve( L.T, np.linalg.solve( L, cov_traintest ) )
    V = cov_test - cov_traintest.T.dot( v )
    # Winv = xtrain.T.dot( np.linalg.solve( cov_train, xtrain ) )
    # A = xtest.T - xtrain.T.dot( v )
    # V += A.T.dot( np.linalg.solve( Winv, A ) )
    
    # Log marginal likelihood
    logp = -1/2*np.dot( ztrain.T, alpha ) - \
           np.sum( np.log( np.diag(L) ) ) - \
           len(xtrain)/2*np.log(2*np.pi)
    
    # Loglikelihood
    logpdf = mvnormal_logpdf( ztrain, mean = m_train, cov = cov_train )
    # logpdf += -1/2*np.log( np.linalg.det(Winv) )

    # V = NoiseCOV( np.exp(params[5]), len(xtest) )
    
    return f, V, logpdf, m_test, S_test

def readPickle( filename, N0=0, offset=1 ):
    
    with open(filename, 'rb') as f:
        sims = pickle.load(f)
        counts = np.array(pickle.load(f) )
        h = np.array(pickle.load(f) )
        elev = np.array(pickle.load(f) )
        array_ext = pickle.load(f)
        array_bdy = pickle.load(f)
        coords = pickle.load(f)
        shore = pickle.load(f)
        array_elev = pickle.load(f)
        
    # Indices of points that have a match
    indices = np.where(np.array(counts)>0)[0]
    
    # Cells where flood height is defined (row,col)
    x = coords[indices]
    x = x[::offset]

    # Flood height in given cells
    z = h[indices]
    z = z[::offset]
    
    # Add non-flooded pixels as observations
    px, py = np.where( array_ext==0 )
    ind = np.random.choice( np.arange(0, len(px), 1), size=N0, replace=False )
    x0 = np.array([px[ind],py[ind]]).T
    # x0 = np.array([[5,10],[5,25],[5,40],[5,55],[6,70],
    #                [15,10],[15,25],[15,40],[15,55],[13,68],
    #                [25,5],[20,25],[20,40],
    #                [35,30],[35,45],[35,55],[35,70],[28,70],
    #                [45,5],[45,25],[45,40],[45,55],[45,70]
    #               ])
    z0 = np.zeros( len(x0) )
    x = np.vstack( [x,x0] )
    z = np.hstack( [z,z0] )
    shore = shore[indices]
    shore = shore[::offset]
    shore = np.hstack( [shore,z0] )
    elev = elev[indices]
    elev = elev[::offset]
    elev0 = np.zeros( len(x0) )
    for i in range(len(x0)):
        elev0[i] = array_elev[ x0[i][0],x0[i][1] ]
    elev = np.hstack( [elev,elev0] )
    
    # Training output dataset
    train = x, z, elev, shore
    
    return train, array_ext, array_bdy, array_elev

