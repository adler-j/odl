# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import print_function, division, absolute_import
import numpy as np

from odl.discr import ResizingOperator
from odl.trafos import FourierTransform, PYFFTW_AVAILABLE


__all__ = ('voxel_driven_compensation_filter')


def _axis_in_detector(geometry):
    """A vector in the detector plane that points along the rotation axis."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    c = np.array([np.vdot(axis, du), np.vdot(axis, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def _rotation_direction_in_detector(geometry):
    """A vector in the detector plane that points in the rotation direction."""
    du, dv = geometry.det_axes_init
    axis = geometry.axis
    det_normal = np.cross(dv, du)
    rot_dir = np.cross(axis, det_normal)
    c = np.array([np.vdot(rot_dir, du), np.vdot(rot_dir, dv)])
    cnorm = np.linalg.norm(c)

    # Check for numerical errors
    assert cnorm != 0

    return c / cnorm


def voxel_driven_compensation_filter(ray_trafo, fu=1.0, fv=1.0, padding=True):
    """Compensate for voxel driven backprojection.

    fx, fy floats in [0, 1]
    """
    impl = 'pyfftw' if PYFFTW_AVAILABLE else 'numpy'

    if ray_trafo.domain.ndim == 2:
        pass
    elif ray_trafo.domain.ndim == 3:
        # Find the direction that the filter should be taken in
        axis = _axis_in_detector(ray_trafo.geometry)
        rot_dir = _rotation_direction_in_detector(ray_trafo.geometry)

        # Define ramp filter
        def fourier_filter(x):
            axis_freq = np.abs(axis[0] * x[1] + axis[1] * x[2])
            rot_dir_freq = np.abs(rot_dir[0] * x[1] + rot_dir[1] * x[2])
            norm_axis_freq_freq = axis_freq / np.max(axis_freq)
            norm_rot_dir_freq = rot_dir_freq / np.max(rot_dir_freq)

            filt = np.exp(-norm_axis_freq_freq ** 2 / fv ** 2
                          -norm_rot_dir_freq ** 2 / fu ** 2)

            return filt

        # Define (padded) fourier transform
        if padding:
            # Define padding operator
            pad_u = int(3 / fu)
            pad_v = int(3 / fv)
            padded_shape_u = ray_trafo.range.shape[1] + pad_u
            padded_shape_v = ray_trafo.range.shape[2] + pad_v

            ran_shp = (ray_trafo.range.shape[0],
                       padded_shape_u,
                       padded_shape_v)
            resizing = ResizingOperator(ray_trafo.range, ran_shp=ran_shp,
                                        pad_mode='order0')

            fourier = FourierTransform(resizing.range, impl=impl)
            fourier = fourier * resizing
        else:
            fourier = FourierTransform(ray_trafo.range, impl=impl)
    else:
        raise NotImplementedError('FBP only implemented in 2d and 3d')

    # Create ramp in the detector direction
    filter_function = fourier.range.element(fourier_filter)

    # Create ramp filter via the convolution formula with fourier transforms
    return fourier.inverse * filter_function * fourier

if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    import odl

    # Create Ray Transform in helical geometry
    reco_space = odl.uniform_discr(
        min_pt=[-20, -20, 0], max_pt=[20, 20, 40], shape=[100, 100, 100])
    geometry = odl.tomo.helical_geometry(reco_space, 500, 500, num_turns=5)
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

    filt = voxel_driven_compensation_filter(ray_trafo,
                                            fu=0.5,
                                            fv=0.1)

    run_doctests()
