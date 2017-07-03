"""Simple example of implemented figure-of-merits (FOMs)

Numerical test of a few implemented FOMs (mean square error, mean
absolute error, density standard deviation, density range, blur, and false
structures) as a function of increasing noise level.

"""

import odl
import odl.contrib.fom
import numpy as np
import matplotlib.pyplot as plt

# Discrete space: discretized functions on the rectangle
# [-20, 20]^2 with 100 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

mse = []
mae = []
mvd = []
std_diff = []
range_diff = []
blur = []
false_struct = []
ssim = []

# Create mask for ROI to evaluate blurring and false structures. Arbitrarily
# chosen as bone in Shepp-Logan phantom.
mask = (np.asarray(phantom) == 1)

for stddev in np.linspace(0.1, 10, 100):
    phantom_noisy = phantom + odl.phantom.white_noise(reco_space,
                                                      stddev=stddev)
    mse.append(odl.contrib.fom.supervised.
               mean_squared_error(phantom_noisy,
                                  phantom,
                                  normalized=True))

    mae.append(odl.contrib.fom.supervised.
               mean_absolute_error(phantom_noisy,
                                   phantom,
                                   normalized=True))

    mvd.append(odl.contrib.fom.supervised.
               mean_value_difference(phantom_noisy,
                                     phantom,
                                     normalized=True))

    std_diff.append(odl.contrib.fom.supervised.
                    standard_deviation_difference(phantom_noisy,
                                                  phantom,
                                                  normalized=True))

    range_diff.append(odl.contrib.fom.supervised.
                      range_difference(phantom_noisy,
                                       phantom,
                                       normalized=True))

    blur.append(odl.contrib.fom.supervised.blurring(phantom_noisy,
                                                    phantom,
                                                    mask,
                                                    normalized=True,
                                                    weight_factor=30))

    false_struct.append(odl.contrib.fom.supervised.
                        false_structures(phantom_noisy,
                                         phantom,
                                         mask,
                                         normalized=True,
                                         weight_factor=30))

    ssim.append(odl.contrib.fom.supervised.
                ssim(phantom_noisy,
                     phantom,
                     normalized=True))

plt.figure()
plt.plot(mse)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Mean square error')

plt.figure()
plt.plot(mae)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Mean absolute error')

plt.figure()
plt.plot(mvd)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Mean value difference')

plt.figure()
plt.plot(std_diff)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Standard deviation difference')

plt.figure()
plt.plot(range_diff)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Range difference')

plt.figure()
plt.plot(blur)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Blurring (weighted importance on foreground)')

plt.figure()
plt.plot(false_struct)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('Blurring (weighted importance on background)')

plt.figure()
plt.plot(ssim)
plt.xlabel('Noise level')
plt.ylabel('FOM')
plt.title('SSIM')
