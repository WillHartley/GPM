from __future__ import print_function,division
import numpy as np
from numpy import exp
import h5py
import sys
import json
import os
import time
from scipy.special import erfinv
import get_fsps_phot as gfp
import yaml
import ghalton

"""
Use a Halton sequence to optimally sample the 6-D param space. Construct a series of hdf5 files 
with model mags and SFH parameters (to use parallelisation).

Call as:
python construct_halton_grid.py <n_objs> <seq_num>

Also requires a yaml config file, "grid.yaml".
"""

def priortransform(cube):
        # This is not used in this code, so don't worry about the fact
        # it doesn't match the other codes.
        # definition of the parameter width, by transforming from a unit cube
        cube = cube.copy()
        cube[0] = cube[0] * 5.5 - 3 # basically, this sets bounds on the mass: 7 to 12.5 -> -3 to 2 (A=1 -> 10^10). Working in log10(mass) units.
        cube[1] = cube[1] * 10 # mu: 0 to 10
        cube[2] = cube[2] * 4.8 + 0.2 # sigma: 0.2 to 5 
        cube[3] = 10**(0.3 * erfinv(2.*(cube[3]-0.5)))  # metallicity scatter: Gaussian about 0, sigma = 0.3 dex
        cube[4] = cube[4] * 2.  # dust extinction (doubled in birth clouds): 0 to 2.
        cube[5] = 10**(cube[5]) - 1. # log-spaced redshift sampling, z: 0 to 9.
        return cube


# read in arguments (use an arg parser, but this'll do for now).
args = sys.argv
n_objs = int(args[1])
seq_num = int(args[2])
print(n_objs, seq_num)    

# read grid parameters from grid.yaml - change to argument?
#config = yaml.load(open(sys.argv[1]))
config = yaml.load(open("grid.yaml"))


## set-up mulit-dim. flux and param cubes
# read in number of bands
nband = config['n_band']


# use a halton sequence to set up the initial param space search
sequencer = ghalton.Halton(6)
points = np.array(sequencer.get(config['total_points']))


# Now we need to select the ones we will actually do
# Halton sequence is deterministic, so we will be generating the same set of points each time.
if n_objs*(1+seq_num) > config['total_points']:
    points = points[n_objs*seq_num:,:]
else:
    points = points[n_objs*seq_num:n_objs*(1+seq_num),:]
#print(points.shape)

#model_grid = np.zeros((points.shape[0],6))
#for i in range(points.shape[0]):
#        model_grid[i,:] = priortransform(points[i,:]) 

# we're going to do the emulation within the unit cube:
model_grid = points

# flux array
fluxes = np.zeros((model_grid.shape[0], nband))
    
# get fluxes for each model grid point
# (can get multiple redshifts efficiently, but not if we want to do
# e-lines on a second pass)
for i in range(model_grid.shape[0]):
    fluxes[i,:] = gfp.get_fsps_fluxes(model_grid[i,:])

# save into hdf5
f = h5py.File(config['out_name']+'_'+str(seq_num)+'.h5', "w")

f.create_dataset("fluxes", fluxes.shape, dtype='f', compression='lzf', data=fluxes)
f.create_dataset("model_params", model_grid.shape, dtype='f', compression='lzf', data=model_grid)

f.close()
