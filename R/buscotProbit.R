library(raster)
library(distr)

library(raster)
library(processx)
library(ggplot2)
library(gridExtra)
library(reshape)
library(tools)
library(data.table)
library(mvtnorm)
library(tlrmvnmvt)
library(pracma)
library(TruncatedNormal)


NoiseCOV <- function( se, n ){
  return( se * eye(n) )
}

GaussianCOV <- function( sd, wx, wy, x_diffs, y_diffs ){
  return( sd * exp( -wx*x_diffs^2 - wy*y_diffs^2 ) )
}

buildCOV <- function( t1, t2x, t2y, s, xy ){
  
  # Total number of variables
  N <- length(xy$X)
  
  # Create distance matrices from grid
  xdist_triang <- as.numeric( dist( xy$X ) )
  ydist_triang <- as.numeric( dist( xy$Y ) )
  cov_triang <- GaussianCOV( t1, t2x, t2y, xdist_triang, ydist_triang )
  
  COV <- matrix( 0, nrow=N, ncol=N )
  COV[lower.tri(COV,diag=FALSE)] <- cov_triang
  COV <- COV + t(COV) # Complete upper triangle
  diag(COV) <- rep(t1,N) # Complete diagonal
  
  rm(xdist_triang, ydist_triang, cov_triang)

  # Add Noise COV
  COV <- COV + s*diag(N)
  
  return( COV )
}

createPar <- function( filename, files, title='', settings=c(), options=c() ){
  
  # Create and write file
  f <- file( filename )
  lines <- c()
  i <- 1
  
  # Title row
  lines[i] <- sprintf('# %s', title)
  i <- i + 1
  lines[i] <- ''
  
  # Optional key-word arguments
  for (j in 1:length(options)){
    i <- i + 1
    name <- names(options)[j]
    value <- options[j]
    lines[i] <- sprintf( '%s\t%s', name, value )
  }
  
  # Files
  for (j in 1:length(files)){
    i <- i + 1
    name <- names(files)[j]
    value <- files[j]
    lines[i] <- sprintf( '%s\t%s', name, value )
  }
  
  # Single-word settings
  i <- i + 1
  lines[i] <- ''
  for (j in 1:length(settings)){
    i <- i + 1
    value <- settings[j]
    lines[i] <- value
  }
  
  writeLines( lines, f )
  close(f)
  
  return()
}

Lisflood <- function( theta, output=NULL ){
  
  # Input raster files
  files <- c()
  files['DEMfile'] <- 'Buscot.dem.asc'
  files['bcifile'] <- 'Buscot.bci'
  files['bdyfile'] <- 'Buscot.bdy'
  files['SGCwidth'] <- 'Buscot.width.asc'
  files['SGCbed'] <- 'Buscot.bed.asc'
  files['SGCbank'] <- files['DEMfile']
  files['startfile'] <- 'Buscot.depth.asc'
  files['weirfile'] <- 'Buscot.weir'
  
  # Settings
  options <- c()
  options['resroot'] <- 'temp'
  options['dirroot'] <- 'temp'
  options['sim_time'] <- 1382400
  options['initial_step'] <- 1
  options['massint'] <- 10*1
  options['saveint'] <- as.numeric(options['sim_time'])/9
  options['SGCn'] <- theta[1]
  options['fpfric'] <- theta[2]
  settings<- c('mint_hk','elevoff')
  
  # Create .par file
  parfile <- 'Buscot.par'
  createPar( parfile, files, settings=settings, options=options )
  
  # Run model
  executable <- './lisflood_linux' #for linux
  # executable <- 'lisflood_intelRelease_double' #for Windows
  aa <- run( executable, parfile, timeout=120, error_on_status=FALSE, 
             env=c('current', OMP_NUM_THREADS=toString(4) ), echo=TRUE )
  
  # Save outputs
  filename <- sprintf('%s.max',options['resroot'])
  source <- file.path( options['dirroot'], filename  )
  if (!is.null(output)){
    file.copy( source, output )
  }
  
  # Extract flood depths array
  src <- raster( source )
  z <- as.matrix( src )
  
  return(z)
  
}

colVars <- function(x, na.rm=FALSE, dims=1, unbiased=TRUE, SumSquares=FALSE,
                    twopass=FALSE) {
  if (SumSquares) return(colSums(x^2, na.rm, dims))
  N <- colSums(!is.na(x), FALSE, dims)
  Nm1 <- if (unbiased) N-1 else N
  if (twopass) {x <- if (dims==length(dim(x))) x - mean(x, na.rm=na.rm) else
    sweep(x, (dims+1):length(dim(x)), colMeans(x,na.rm,dims))}
  (colSums(x^2, na.rm, dims) - colSums(x, na.rm, dims)^2/N) / Nm1
}

logPrior <- function( logpriors, params ){
  
  lps <- array(0,length(logpriors))
  for (i in 1:length(logpriors)){
    lps[i] <- d( logpriors[[i]] )( params[i], log=TRUE  )
  }
  return( sum(lps) )
}

logLikelihood <- function( z, params, xy, output='' ){
  
  # Number of observations
  N <- length(z)
  
  # Run Lisflood
  S <- Lisflood( exp(params[1:2]), output=output )
  S_flat <- S[ cbind(xy$Y,xy$X) ]
  # if (bounds != 'all'){
  #   S <- S[bounds[1]:bounds[2],bounds[3]:bounds[4]]
  # }
  # S_flat <- as.vector( t(S) )
  
  # Compute COV
  start_time <- Sys.time()
  # Create grid
  # grid <- meshgrid( seq(from = 0, by = 1, l = ncol(z)),
  #                   seq(from = 0, by = 1, l = nrow(z)) )
  COV <- buildCOV( exp(params[3]), exp(params[4]), exp(params[4]),
                   exp(params[5]), xy )
  end_time <- Sys.time()
  dt <- end_time - start_time
  print( sprintf('COV built in %fs',dt) )
  
  # Probit log-likelihood
  start_time <- Sys.time()
  lower <- rep(0,N)
  upper <- rep(0,N)
  for (i in 1:N){
    if (z[i]==0){
      lower[i] = -Inf
      upper[i] = 0
    } else if (z[i]==1){
      lower[i] = 0
      upper[i] = Inf
    }
  }

  L <- TruncatedNormal::pmvnorm( lb=lower, ub=upper, mu=S_flat, sigma=COV,
                                 type='mc', B=5000 )
  logL <- log(L[1])

  
  end_time <- Sys.time()
  dt <- end_time - start_time
  print( sprintf('Log Likelihood computed in %fs',dt) )
  
  # Remove variables
  rm(S,S_flat,lower,upper)
  
  return( logL )
}

logPosterior <- function( z, logpriors, params, xy, output ){
  lprior <- logPrior( logpriors, params )
  if (lprior == -Inf){
    return( -Inf )
  }
  logL <- logLikelihood( z, params, xy, output ) + lprior
  return( logL )
}


plot_diagnostics <- function( Xpaths, R, logpriors ){
  
  # Number of variables
  Nvars <- length(Xpaths[1,,1])
  Nsim <- length(Xpaths[,1,1])
  count <- 0
  
  plist <- list()
  for (i in 1:Nvars){
    
    # Plot of densities
    count <- count + 1
    
    # Create data frame
    df <- as.data.frame( Xpaths[,i,] )
    df['id'] <- 1:Nsim
    dprior <- d( logpriors[[i]] )( df$id )
    df <- melt( df, id='id' )
    
    # Plot densities for each path
    plist[[count]] <- ggplot( data=df, aes(x=value, colour=variable) )
    plist[[count]] <-  plist[[count]] + geom_density(show.legend = FALSE)
    
    # Add prior density
    sep <- abs(max(df$value) - min(df$value))/200
    xprior <- seq(from=min(df$value), to=max(df$value), by=sep)
    dprior <- d( logpriors[[i]] )( xprior )
    df_prior <- data.frame( prior=dprior, x=xprior )
    plist[[count]] <- plist[[count]] + geom_line( data=df_prior, aes(x=x,y=prior),
                                                  inherit.aes = FALSE,
                                                  linetype = "dashed")
    
    # Plot of series
    count <- count + 1
    
    # Plot densities for each path
    plist[[count]] <- ggplot( data=df, aes(x=id, y=value, colour=variable) )
    plist[[count]] <- plist[[count]] + geom_line(show.legend = FALSE)
    
    # Plot of autocorrelation
    count <- count + 1
    
    # Plot densities for each path
    acorr <- data.frame( acf=acf( df$value, lag.max = Nsim-1, plot = FALSE)$acf[,1,1] )
    acorr['id'] <- 1:Nsim
    plist[[count]] <- ggplot( data=acorr, aes(x=id, y=acf) ) + geom_bar(stat='identity')
    plist[[count]] <- plist[[count]] + xlim(0,200)
    
  }
  
  # Plot paths Densities
  g <- grid.arrange(grobs = plist, ncol = 3) ## display plot
  print(g) # to show it when sourcing
  
  return()
}

mh_sampler <- function( vars, target_logpdf, Nsim, x0, jump_S, burnin=0, 
                        resultscsv='mh_summary.csv' ){
  
  # Number of parameters
  Nvars <- length( x0 )
  
  # Flood file names
  lfpname <- file_path_sans_ext( resultscsv )
  
  # Create data base
  f <- file( resultscsv )
  
  # Header
  header <- as.list( c( vars, c('r', 'accepted', 'logpdf', 'mapname') ) )
  fwrite( header, file=resultscsv, append=FALSE )
  close(f)
  
  # Run paths
  X <- array( 0, c(Nsim, Nvars) )
  acceptance <- array( 0, Nsim )
  x_prev <- x0
  output <- sprintf('%s_sim0.max',lfpname)
  logpdf_prev <- target_logpdf( x_prev, output=output )
  line <- as.list( c(x0, c('-','-',logpdf_prev,output)) )
  fwrite( line, file=resultscsv, append=TRUE )
  
  # Iterations
  for (i in 1:Nsim){
    
    # Sample proposal x from jumping distribution
    x_prop <- rmvnorm( 1, mean=x_prev, sigma=jump_S )
    
    # Calculate loglikelihood for proposed x
    output <- sprintf('%s_sim%d.max',lfpname,i)
    logpdf_prop <- target_logpdf( x_prop, output=output )
    
    # Calculate ratio of densities
    r <- (logpdf_prop - logpdf_prev)
    r <- r + dmvnorm(x_prop, mean=x_prev, sigma=jump_S, log=TRUE)
    r <- r - dmvnorm(x_prev, mean=x_prop, sigma=jump_S, log=TRUE)
    r <- exp(r)
    
    # Accept with probability r
    if (runif(1) <= min(r,1)){
      x_new <- x_prop
      logpdf_prev <- logpdf_prop
      acceptance[i] <- 1
    } else {
      x_new <- x_prev
      acceptance[i] <- 0
    }
    
    # Update data frame for next iteration
    X[i,] <- x_new
    x_prev <- x_new
    
    # Write in csv
    line <- as.list( c( x_prop, c( r, acceptance[i], logpdf_prop, output ) ) )
    fwrite( line, file=resultscsv, append=TRUE )
    
  }
  
  # Remove burnin samples
  Xbin <- X[(burnin+1):Nsim,]
  
  # Acceptance rate
  acc_rate <- cumsum(acceptance) / (1:Nsim)
  acc_rate <- acc_rate[(burnin+1):Nsim]
  
  return( list('X'=X, 'Xbin'=Xbin, 'acc_rate'=acc_rate) )
}

mh_adaptive_sampler <- function( vars, target_logpdf, Nsim, x0, jump_S,
                                 tune, tune_intvl, burnin, resultscsv){
  
  # Number of parameters
  Nvars <- length(x0)
  
  # Initial covariance matrix
  S <- jump_S
  
  # Initial scaling factor
  gamma <- 1
  
  # Run intervals of tuning period
  for (i in 1:as.integer(tune/tune_intvl)){
    
    # Normal MH sampling
    mh <- mh_sampler( vars, target_logpdf, tune_intvl, x0, gamma*S,
                      burnin=0, resultscsv=resultscsv )
    X <- mh$X
    Xbin <- mh$Xbin
    acc_rate <- mh$acc_rate
    
    # Update jumping covariance
    S_new <- cov(X)
    if (det(S_new)<1e-10){
      S <- S
    } else {
      S <- S_new
    }
    
    # Update new point
    N <- length(X[,1])
    x0 <- X[N,]
    
    if (acc_rate[N] < 0.001){
      gamma <- gamma*0.1
    } else if (acc_rate[N] < 0.05){
      gamma <- gamma*0.5
    } else if (acc_rate[N] < 0.2){
      gamma <- gamma*0.9
    } else if (acc_rate[N] > 0.95){
      gamma <- gamma*10
    } else if (acc_rate[N] > 0.75){
      gamma <- gamma*2
    } else if (acc_rate[N] > 0.5){
      gamma <- gamma*1.1
    }
    
  }
  
  # Simulation phase
  mh <- mh_sampler( vars, target_logpdf, Nsim, x0, gamma*S, burnin, 
                    resultscsv=resultscsv )
  
  return( mh )
}

paths_diagnostics <- function( Xpaths, plot=FALSE, logpriors=c() ){
  
  # Compute mixing and convergences for each variable
  aux <- dim(Xpaths)
  n <- aux[1] # Number of samples per chain
  Nvars <- aux[2] # Number of variables
  if (length(aux)>2){# Number of chains
    m <- aux[3]
  } else {
    m <- 1
  }
  
  # Within-path means
  phat_j <- colMeans( Xpaths, dims=1 )
  
  # Between-paths variance
  B <- n * apply( phat_j, 1, var )
  
  # Within-paths variance
  sj2 <- colVars( Xpaths, dims=1 )
  
  # Within-paths mean
  W <- apply( sj2, 1, mean )
  
  # Marginal posterior variance of estimand
  var_j <- (n-1)/n * W + 1/n * B
  
  # Potential scale reduction (Gelman-Rubin coefficient)
  R <- sqrt( var_j/W )
  
  # Variogram and correlation
  max_lag <- n
  Vt <- array( 0, c(n-1, Nvars) )
  for (t in 1:(n-1)){
    Vti <- apply( colMeans( (Xpaths[1:(n-t+1),,] - Xpaths[t:n,,])^2, dims=1 ), 1, mean)
    Vt[t,] <- Vti
  }
  
  # Autocorrelation
  rhot <- 1 - Vt/2/var_j
  
  # Effective number of samples
  aux <- rhot[2:(n-1),] + rhot[1:(n-2)]
  T <- array( 0, Nvars )
  neff <- array( 0, Nvars )
  for (i in 1:Nvars){
    indices <- match( TRUE, aux[,i]<0 )
    if (!is.na(indices)){
      T[i] <- indices
    } else if (is.na(indices)){
      T[i] <- -1
      neff[i] <- 0
      next
    }
    # Effective number of samples
    neff[i] <- n*m/(1+2*sum(rhot[1:T[i],i]))
  }
  
  # Plot
  if (plot){
    plot_diagnostics( Xpaths, R, logpriors )
  }
  
  return( list("R"=R, "var_j"=var_j, "rhot"=rhot, "neff"=neff) )
  
}

mh_paths_sampler <- function( vars, target_logpdf, Npaths, Nsim, x0, jump_S,
                              burnin=burnin, thin=thin, sampler=mode, tune=tune,
                              tune_intvl=tune_intvl, outputDir=outputDir ){
  
  # Number of parameters
  Nvars <- length( x0[1,] )
  
  #Check that Npaths and x0 are the same dimension
  if ( length(x0[,1]) < Npaths ){
    return( list(-1,-1,-1,-1) )
  }
  
  # Run paths
  X <- array( 0, c(Nsim, Nvars, Npaths) )
  Xbin <- array( 0, c(Nsim-burnin, Nvars, Npaths) )
  acc_rate <- array( 0, c(Nsim-burnin, Npaths) )
  for ( i in 1:Npaths ){
    
    print( sprintf( 'Starting path %d of %d', i, Npaths ) )
    
    # Results for current path
    filename <- file.path( outputDir, sprintf('Path%d_results.csv',i) )
    # Run MCMC scheme according to sampler mode
    if (sampler=='normal'){
      mh <- mh_sampler( vars, target_logpdf, Nsim, x0[i,], jump_S, burnin, 
                        resultscsv=filename )
      X[,,i] <- mh$X
      Xbin[,,i] <- mh$Xbin
      acc_rate[,i] <- mh$acc_rate
    } else if (sampler=='adaptive'){
      mh <- mh_adaptive_sampler( vars, target_logpdf, Nsim, x0[i,], jump_S,
                                 tune=tune, tune_intvl=tune_intvl, 
                                 burnin=burnin, resultscsv=filename )
      X[,,i] <- mh$X
      Xbin[,,i] <- mh$Xbin
      acc_rate[,i] <- mh$acc_rate
    }
    
  }
  
  # Remove trivial paths
  keep_indices <- c()
  for (i in 1:Npaths ){
    path <- Xbin[,1,i]
    if ( length(unique(path))>1 ){ # If all elements are not the same
      keep_indices <- c( keep_indices, i )
    }
  }
  Xbin <- Xbin[,,keep_indices]
  
  # Compute mixing and convergence properties of the chains
  diagnostics <- paths_diagnostics( Xbin, plot=FALSE )
  
  # Thinning of paths
  Xbin <- Xbin[seq(1, length(Xbin[,1,1]), thin),,]
  
  # Stack paths
  Xstack <- Xbin[,,1]
  for (i in 2:Npaths){
    xpath <- Xbin[,,i]
    Xstack <- rbind( Xstack, xpath )
  }
  
  return( list("Xbin"=Xbin, "Xstack"=Xstack, "acc_rate",
               "diagnostics"=diagnostics) )
  
}

# Data (binary flood extent)
bounds <- c(10,45,1,68)

observed <- 'Observations//BuscotFlood92_0.tiff'
src <- raster( observed )
z <- as.matrix( src )
zrev <- apply(z[bounds[1]:bounds[2],bounds[3]:bounds[4]],2,rev)
image(t(zrev))

# Observed points
grid <- meshgrid( seq(from = bounds[3], by = 1, l = (bounds[4]-bounds[3]+1)),
                  seq(from = bounds[1], by = 1, l = (bounds[2]-bounds[1]+1)) )
xy <- c()
xy$X <- as.vector( t(grid$X) ) #cols
xy$Y <- as.vector( t(grid$Y) ) #rows
xy$z <- z[ cbind( xy$Y,xy$X) ]
df <- na.omit( data.frame(xy) )

# Prior of model parameters
vars <- c('r_ch', 'r_fp', 't1', 't2', 'se' )
logrch_prior <- distr::Norm( mean=log(0.03), sd=0.1 ) # Channel
logrfp_prior <- distr::Norm( mean=log(0.045), sd=0.1 ) # Floodplain
logt1_prior <- distr::Norm( mean=-3, sd=0.01 )
logt2_prior <- distr::Norm( mean=-2, sd=1 )
logse_prior <- distr::Norm( mean=-2, sd=0.01 )
logpriors <- c( logrch_prior, logrfp_prior, logt1_prior, logt2_prior, logse_prior )

# MCMC options
outputDir <- 'results/Probit_full_060722_i9'
mode <- 'adaptive'
tune <- 3000
tune_intvl <- 500
resultscsv <- file.path( outputDir, 'mh_summary.csv' )
dir.create( outputDir, showWarnings = TRUE )

# Target distribution
target_logpdf <- function( params, output ){
  return( logPosterior( df$z, logpriors, params, df, output ) )
}

# Jump distribution
sigmas <- c(0.02, 0.02, 0.003, 0.1, 0.003)
jump_S <- sigmas^2 * eye( length(sigmas) )

# Metadata file
md_file <- file.path( outputDir, 'metadata.txt' )
f <- file( md_file )
lines <- c( sprintf('Mode: %s (tune: %d, intvl: %d) ', mode, tune, tune_intvl) )
lines <- c( lines, sprintf('Initial Sigmas: %s', toString(sigmas)) )
lines <- c( lines, sprintf('LogPriors: ') )
for (i in 1:length(sigmas)){
  lines <- c( lines, sprintf('%s \t %f \t %f', class(logpriors[[i]])[1],
                             mean(logpriors[[i]]), sd(logpriors[[i]]))  )
}
lines <- c( lines, sprintf('bounds: %d:%d , %d:%d', bounds[1], bounds[2], bounds[3], bounds[4]) )
lines <- c( lines, sprintf('sf=0.1, sd=0.1'))
writeLines( lines, f )
close(f)

# MH sampler
Nsim <- 10000
burnin <- as.integer(Nsim/2)
Npaths <- 2
thin <- 1
x0 <- array( 0, c(Npaths+1, length(sigmas)) )
for (i in 1:length(sigmas) ){
  x0[,i] <- r(logpriors[[i]])( Npaths+1 )
}
x0[1,4] <- -2.5
x0[2,4] <- -1.5
res <- mh_paths_sampler( vars, target_logpdf, Npaths, Nsim, x0, jump_S,
                         burnin=burnin, thin=thin, sampler=mode, 
                         tune=tune, tune_intvl=tune_intvl,
                         outputDir=outputDir)





