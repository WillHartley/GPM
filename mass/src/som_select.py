from __future__ import print_function
import numpy as np
import h5py
from mvpa2.suite import *
import cPickle as pickle
import random_forest_flux as rff
import sys

"""
Purpose is to use a SOM to split the galaxy sample into similar SEDs, which can then be fed into the collaborative multinest for maximum efficiency. Of course a SOM implementation has its uses anyway...
"""

def hit_map(data):
    # data has already been mapped, and should be 2-d array (ndata, 2)
    x_dim = np.max(data[:,0])
    y_dim = np.max(data[:,1])
    hits = np.zeros((x_dim,y_dim))
    for i in range(x_dim):
        for j in range(y_dim):
            hits[i,j] = ((data[:,0]==i)&(data[:,1]==j)).sum()
    return hits

def write_sub_file(x, y, mapped, i, j, phot_file):
    row_list = np.where((mapped[:,0]==i)&(mapped[:,1]==j))[0]
    oname = 'subfiles/'+phot_file.split('.')[0]+'_sub'+str(i)+'_'+str(j)+'.h5'
    f = h5py.File(oname,'w')
    f.create_dataset('x', data=x)
    f.create_dataset('y', data=y[:,row_list])
    f.create_dataset('row_nums', data=row_list)
    f.close()


# ============================================================
phot_file = 'UDS_DR11_photom.h5'
map_dims = (30,30)

depths = [26.75, 27.6, 27.2, 27.0, 27.0, 26.0, 24.6, 25.6, 25.1, 25.3, 24.2, 24.0]
depths += 2.5*np.log10(5.)

# load data
with h5py.File(phot_file, 'r') as f:
    x = np.array(f['x'].value)
    y = np.array(f['y'].value)
    flags = np.array(f['flags'].value)
ndata = y.shape[1]
flux = y[:y.shape[0]/2,:]
errs = y[y.shape[0]/2:,:]
nbands = flux.shape[0]

# use random forest to fill in fluxes where they are missing
# (purely for the purpose of assigning a SOM cell)
mags = rff.fill_flux(flux, flags) # prob. better using the flags (FLAG4)
#sys.exit()

# compute mags - don't need, the RF retuns mags
#mags = 30.-2.5*np.log10(flux)

# place negative fluxes at 1sigma depth
#for i in range(nbands):
#    mags[i,np.isnan(mags[i,:])] = depths[i]

# negative fluxes have already been placed at 99., so change those to depth
for i in range(nbands):
    mags[i,mags[i,:]>90.] = depths[i]
    
# set-up the SOM and train
som = SimpleSOMMapper(map_dims, 400, learning_rate=0.05)
som.train(mags.T)

# map magnitudes to cells
mapped = som(mags.T)

# save all 900 sub-files
for i in range(map_dims[0]):
    for j in range(map_dims[1]):
        write_sub_file(x, y, mapped, i, j, phot_file)

# get SEDs for each cell and pickle the result
map_cols = np.zeros((map_dims[0],map_dims[1],nbands))
for i in range(map_dims[0]):
    for j in range(map_dims[1]):
        map_cols[i,j,:] = som.reverse1([i,j])
pickle.dump(map_cols, open( "map_cols.p", "wb" ) )
# map_cols = pickle.load( open( "map_cols.p", "rb" ) )
