"""Simple example of implemented figure-of-merits (FOMs)

Numerical test of a few implemented FOMs (mean square error, mean
absolute error, blur, and false structures) as a function of
increasing noise level.

"""

import odl
import odl.contrib.fom
import numpy as np
import matplotlib.pyplot as plt

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

mse = []
mae = []
blur = []
false_struct = []

mask = (np.asarray(phantom) == 1)

for i in np.linspace(0.1, 1, 10):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space, stddev=i)
    mse.append(odl.contrib.fom.mean_square_error(phantom_noisy, phantom))
    mae.append(odl.contrib.fom.mean_absolute_error(phantom_noisy, phantom))
    blur.append(odl.contrib.fom.blurring(phantom_noisy, phantom, mask))
    false_struct.append(odl.contrib.fom.false_structures(phantom_noisy,
                                                         phantom, mask))

plt.plot(mse)
