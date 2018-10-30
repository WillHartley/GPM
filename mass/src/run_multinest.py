import numpy as np
import h5py
import sys
import yaml
import generate_emulator as ge
import gpflow
import astropy.io.fits as pyfits # change to pandas table
import time
import matplotlib.pyplot as plt
import corner
import os
from scipy.special import erfinv
from scipy.special import erf
from pymultinest.solve import solve
import pymultinest as mn
import time


# One possibility here is to use the mean flux in the SOM cell for an initial multinest.
# Then run a further ~ 10 (drawn evenly from the ordered similarity list) to explore at what point the 
# nesting diverges between objects.
# we can then start each object at the point where they break off from one another.

start = time.time()
if not os.path.exists("chains"): os.mkdir("chains")

def priortransform(cube):
    # definition of the parameter width, by transforming from a unit cube - switched redshift to be the last index
    # skip this, emu on unit cube
    #cube = cube.copy()
    cube[0] = cube[0] * 5.5 + 7.
    ##cube[0] = 10**(cube[0] * 5.5 - 3) # basically, this sets bounds on the mass: 7 to 12.5 -> -3 to 2 (A=1 -> 10^10)
    cube[1] = cube[1] * 10 # mu: 0 to 10
    cube[2] = cube[2] * 4.8 + 0.2 # sigma: 0.2 to 5 
    ##cube[3] = cube[3] * 10 # z: 0 to 10
    cube[3] = 10**(0.3 * erfinv(2.*(cube[3]-0.5))) # metallicity scatter: Gaussian about 0, sigma = 0.3 dex - pinch this at some limiting values - 0.001, 0.999??
    cube[4] = cube[4] * 2.  # dust extinction (doubled in birth clouds): 0 to 2.
    cube[5] = 10**(cube[5]) - 1. # log-spaced redshift sampling, z: 0 to 9.
    ##cube[5] = cube[5] # z: 0 to 9. The log spacing is set earlier, so we just output the same value
    return cube
    
def untransform(cube):
    cube[0] = (cube[0] -7.) /  5.5
    cube[1] = cube[1] / 10 # mu: 0 to 10
    cube[2] = (cube[2] - 0.2) / 4.8 # sigma: 0.2 to 5 
    cube[3] = erf(np.log10(cube[3]) / 0.3)/2. + 0.5 # metallicity scatter: Gaussian about 0, sigma = 0.3 dex - pinch this at some limiting values - 0.001, 0.999??
    cube[4] = cube[4] / 2.  # dust extinction (doubled in birth clouds): 0 to 2.
    cube[5] = np.log10(cube[5] + 1.) # log-spaced redshift sampling, z: 0 to 9.
    return cube

def mnest_prior(cube, ndim, nparams):
    return cube

def gauss(x, z, A, mu, sig):
    return A * exp(-0.5 * ((mu - x / (1. + z))/sig)**2)

def get_mod_phot(A, mu, sig, met, dust, z):
    y, yerr = mod.predict_y(np.array([A, mu, sig, met, dust, z]).reshape(1,-1))
    return 10**((30.-y[0,:12])/2.5)




# pick which object (for testing)
i_obj = 48

# read yaml
config = yaml.load(open("grid.yaml"))
    
# read in the emulator, compute it if needed. Pickling the object doesn't seem to work?
if config['train_emu'] == True:
    mod = ge.make_emulator_GPflow()
else:
    mod = ge.load_emulator_GPflow()

# load data
ndata = 100
if ndata > 0:
    with h5py.File("UDS_DR11_spec.h5", 'r') as f:
        x = np.array(f['x'].value)
        y = np.array(f['y'][:,:ndata])
else:
    with h5py.File("UDS_DR11_spec.h5", 'r') as f:
        x = np.array(f['x'].value)
        y = np.array(f['y'].value)

nx, ndata = y.shape
# y has fluxes followed by flux errors.

# we want to add systematic errors to the photometry, by band:
bands = ['u','B','V','R','i','z','Y','J','H','K','I1','I2']
syst_errs = np.array([0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.1, 0.1])
y[int(nx/2):,:] = np.sqrt((y[int(nx/2):,:]*y[int(nx/2):,:]) + (np.tile(syst_errs, ndata).reshape((ndata,int(nx/2))).T * y[:int(nx/2),:])**2)
waves = np.array([3796.11571888, 4458.05618762, 5477.41955513, 6532.37444814, 7682.94106106, 9036.67456649, 10212.72604114, 12510.43831574, 16387.79389646, 22084.84812739, 35572.60440514, 44976.10885919])


params = ['A', 'mu', 'sig', 'met', 'tau', 'z']
nparams = len(params)


print(y.shape)
# length of data vector
n = len(y[:int(nx/2),i_obj])


fluxes = y[:int(nx/2),i_obj]
flux_errs = y[int(nx/2):,i_obj]
print(fluxes)


#labels = ['u', 'B', 'V', 'R', 'i', 'z', 'Y', 'J', 'H', 'K', 'I1', 'I2']
#fig = plt.figure(figsize=(14,6))
#ax = fig.add_subplot(111)
#ax.plot(waves,fluxes/waves,'o',c='blue')
#ax.errorbar(waves,fluxes/waves,yerr=flux_errs/waves,c='blue',fmt='o')
#ax.set_xlabel('Wavelength')
#ax.set_ylabel(r"${\rm Flux}~\nu~f_{\nu}$")
#ax.set_yscale('log')
#for i,j,k in zip(waves,fluxes,labels):
#    ax.annotate('%s' %k, xy=(i,j), xytext=(-10,10), textcoords='offset points')
#ax.legend()


# now we start the multinest stuff
def loglike(params, ndim, nparams, lnew):
    #A, mu, log_sig_kms = params
    # predict the model
    # call external fn to generate FSPS spec, photometry, inc. Madau extinct., save mass
    #ypred = gfp.get_fsps_phot(params) # params are log-normal + redshift
    #sig = 10**log_sig_kms
    #ypred = A * exp(-0.5 * ((mu - x)/sig)**2)
    # do the data comparison - data is set-up as fluxes, errors, so second half of data are the uncert.
    ypred = get_mod_phot(params[0], params[1], params[2], params[3], params[4], params[5])
    #print(ypred.shape, ypred)
    # convert mags to fluxes, fluxes = 10**((30. - mags)/2.5)
    L = -0.5 * (((ypred - fluxes) / flux_errs)**2).sum()
    return L


prefix = "chains/test5_"


# run MultiNest
result = mn.run(loglike, mnest_prior, nparams, outputfiles_basename=prefix, verbose=True, resume=False)
#result = solve(loglike, mnest_prior)


print("Done in, ",time.time()-start," seconds")


#print(result)

