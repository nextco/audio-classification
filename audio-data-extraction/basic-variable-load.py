# https://docs.scipy.org/doc/numpy/reference/generated/numpy.load.html
import numpy as np


data = np.load('variable-save.npz')
print(data) # umpy.lib.npyio.NpzFile

samples_x = data['samples_x']
samples_y = data['samples_y']

print(samples_x)
print(samples_y)
