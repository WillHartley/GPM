from __future__ import print_function
import h5py
import astropy.io.fits as pyfits
import numpy as np
from astropy.table import Table
import sys

# need to decide whether to include zero-point adjustments in the masses....

if len(sys.argv) != 3:
    print('use: python make_hdf.py input(fits)file output(hdf5)file')
    sys.exit()

infile = sys.argv[1]
outfile = sys.argv[2]

#bands = ['u','B','V','R','i','z','newz','Y','J','H','K','1','2']
bands = ['u','B','V','R','i','z','Y','J','H','K','I1','I2']

# read in fits file
t = Table.read(infile)

# x are just labels - should be fine!
x = np.tile(np.array(bands),2)

# construct data array
y = np.empty((2*len(bands),len(t['NUMBER'])))

for i,band in enumerate(bands):
    col = band+'FLUX_20'
    y[i,:] = t[col]
    col = band+'FLUXERR_20'
    y[i+len(bands),:] = t[col]

flagid = np.vstack((t['NUMBER'],t['FLAG4']))
print(flagid.shape, y.shape)

try:
    z_spec = t['z_spec']
except:
    z_spec = np.zeros(len(t['NUMBER']))
    
# now open and write the hdf5 file
f = h5py.File(outfile,'w')
f.create_dataset('x', data=x.astype('S8'))
f.create_dataset('y', data=y)
f.create_dataset('flags', data=flagid)
f.create_dataset('z_best', data=z_spec)
f.close()
