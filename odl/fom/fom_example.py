"""Simple test of implemented figure-of-merits (FOMs)

Numerical test of a few implemented FOMs (mean square error, mean
absolute error, blur, and false structures) as a function of
increasing noise level.

"""

import odl
import numpy as np
#import fom
import matplotlib.pyplot as plt

# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 558)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection). We use the 'skimage' backend.
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

mse = []
mae = []
blur = []
false_struct = []

mask = (np.asarray(phantom) == 1)

for i in np.linspace(0.1, 1, 10):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space, stddev=i)
    mse.append(odl.fom.mean_square_error(phantom_noisy, phantom))
    mae.append(odl.fom.mean_absolute_error(phantom_noisy, phantom))
    blur.append(odl.fom.blurring(phantom_noisy, phantom, mask))
    false_struct.append(odl.fom.false_structures(phantom_noisy, phantom, mask))
plt.plot(mse)
