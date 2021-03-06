# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Example on using show and updating the figure in real time in 2d."""

import odl
import matplotlib.pyplot as plt

n = 100
m = 20
space = odl.uniform_discr([0, 0], [1, 1], [n, n])
phantom = odl.phantom.shepp_logan(space, modified=True)

# Create a figure by saving the result of show
fig = None

# Reuse the figure indefinitely, values are overwritten.
for i in range(m):
    fig = (phantom * i).show(fig=fig, clim=[0, m])
    plt.pause(0.1)

plt.show()
