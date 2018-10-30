from __future__ import print_function, division
import numpy as np
import h5py
import sklearn.gaussian_process as gp
import yaml
import pickle
import gpflow
import tensorflow as tf

"""
This generates the Gaussian process emulator.
By default it uses the fluxes (actually, magnitudes) from the flux file defined in grid.yaml.
However, it can be called with an argument that contains additional evaluations
in arrays with the same format as the HDF5 file. (need to implement this).
"""

def read_sps_fluxes(hdf5file):
    with h5py.File(hdf5file, 'r') as h5:
        fluxes = np.array(h5['fluxes'].value)
        par_cube = np.array(h5['model_params'].value)
        print(fluxes.shape, par_cube.shape)
    return fluxes, par_cube


def make_emulator_GPflow(extra_models=None):

    # read in yaml file
    config = yaml.load(open("grid.yaml"))
    
    # read in fluxes, and add any extra ones (not implememnted yet).
    ypred, pars = read_sps_fluxes(config['out_name']+'.h5')
    print(ypred.shape, pars.shape, pars.dtype, ypred.dtype)

    # change data type for tensor flow
    ypred = np.array(ypred, dtype=np.float64)
    pars = np.array(pars, dtype=np.float64)

    # https://github.com/GPflow/GPflow/issues/689
    with gpflow.defer_build():
        
        # define the covariance kernel
        k = gpflow.kernels.Matern52(input_dim=6, ARD=True, lengthscales=config['l_scl'])

        # generate the model
        mod = gpflow.models.GPR(pars, ypred, kern=k, name="mod")

    tf.local_variables_initializer()
    tf.global_variables_initializer()

    tf_session = mod.enquire_session()
    mod.compile(tf_session)

    # fit the GP to the computed SFHs
    gpflow.train.AdagradOptimizer(0.1).minimize(mod)

    # save the model
    saver = tf.train.Saver()
    save_path = saver.save(tf_session, "./model.ckpt")
    
    return mod


def load_emulator_GPflow():

    # read in yaml file
    config = yaml.load(open("grid.yaml"))

    # read in fluxes, and add any extra ones (not implememnted yet).
    ypred, pars = read_sps_fluxes(config['out_name']+'.h5')
    print(ypred.shape, pars.shape, pars.dtype, ypred.dtype)

    # change data type for tensor flow
    ypred = np.array(ypred, dtype=np.float64)
    pars = np.array(pars, dtype=np.float64)

    with gpflow.defer_build():
        
        # define the covariance kernel
        k = gpflow.kernels.Matern52(input_dim=6, ARD=True, lengthscales=config['l_scl'])

        # generate the model
        mod = gpflow.models.GPR(pars, ypred, kern=k, name="mod")

    tf_graph = mod.enquire_graph()
    tf_session = mod.enquire_session()
    mod.compile(tf_session)

    saver = tf.train.Saver()

    save_path = saver.restore(tf_session, "./model.ckpt")

    return mod
    

def make_emulator(extra_models=None):

    # read in yaml file
    config = yaml.load(open("grid.yaml"))
    
    # read in fluxes, and add any extra ones (not implememnted yet).
    ypred, pars = read_sps_fluxes(config['out_name']+'.h5')
    print(ypred.shape, pars.shape)

    # define the covariance kernel
    kern = gp.kernels.Matern(length_scale=config['l_scl'])

    # generate the model
    mod = gp.GaussianProcessRegressor(kernel=kern, alpha=1.e-2, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=100, normalize_y=True, copy_X_train=True, random_state=10)

    # fit the GP to the computed SFHs
    mod.fit(pars,ypred)

    return mod

    # pickle dump the fit model. - doesn't seem to work, try shelving
    #file_pick = open(config['emu_file'], 'wb')
    #pickle.dump(mod, file_pick)
    #file_pick.close()

"""
Junk from notebook:

from __future__ import print_function, division
import numpy as np
import GPy
import h5py
%matplotlib inline
from matplotlib import pyplot as plt
import random
import sklearn.gaussian_process as gp

# looks like we might get more success if we work with magnitudes. :\
def read_sps_fluxes(hdf5file):
with h5py.File(hdf5file, 'r') as h5:
    fluxes = np.array(h5['fluxes'].value)
    par_cube = np.array(h5['par_cube'].value)
    return fluxes, par_cube

# read in precomputed fluxes
ypred, pars = read_sps_fluxes('flux_hdf5_2.h5')
print(ypred.shape, pars.shape)

# cut this down to our first test set: met (1.), tau(0.), z(0.6 - 0.8) 
# - marg over SFH params, something odd, all are zero... damn... (smaller run was wrong in a different way)
# something wrong with how I set up / save the cube. Doesn't matter. 
# &(pars[:,0]==1.2)&(pars[:,1]==1.2)
subset = np.where((pars[:,4]>=0.)&(pars[:,4]<=10.)&(pars[:,2]==1.)&(pars[:,3]==0.)&(pars[:,0]==1.2))[0]
print(len(subset))
print(np.unique(pars[subset,1]))
Y = 30.-2.5*np.log10(ypred[subset,:])
#Y = (ypred[subset,9])
#Y = Y.reshape(len(Y),1)
X = pars[subset,4]
#X = X[:,np.array((0,4))]
X = X.reshape(len(X),1)
print(X.shape, Y.shape)
#print(Y)

# randomly drop some points
keep = np.random.choice(np.arange(len(Y)), size=int(len(Y)/20.))
Y = Y[np.array(keep,dtype=int),:]
X = X[np.array(keep,dtype=int),:]

x_pred = np.linspace(0, 10, 5000)[:,None]


# Now let's try the scikit version
# gp.kernels.ExpSineSquared()
# gp.kernels.Sum(gp.kernels.ConstantKernel(),gp.kernels.ExpSineSquared())
kern = gp.kernels.Matern(length_scale=2.)
#kern = gp.kernels.Sum(gp.kernels.Matern(length_scale=2.),gp.kernels.WhiteKernel())
mod = gp.GaussianProcessRegressor(kernel=kern, alpha=1e-10, optimizer='fmin_l_bfgs_b', 
                                  n_restarts_optimizer=100, normalize_y=True, copy_X_train=True, random_state=10)

mod.fit(X,Y)

pred_sci, std_sci = mod.predict(x_pred, return_std=True)
print(pred_sci.shape)


    
"""
