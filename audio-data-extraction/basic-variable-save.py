# https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html
import numpy as np

# Pruebas
samples_x = np.matrix('0 1 2 4 5 6 7 8 9;'
                   '3 4 5 6 4 5 4 8 0;'
                   '1 5 5 15 4 5 4 8 0')

print(samples_x)

# Etiquetas
samples_y = np.array([0, 1, 2])

np.savez('variable-save', samples_x=samples_x, samples_y=samples_y)
