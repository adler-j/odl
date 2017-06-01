# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""
Implementation of Figure of Merits (FOMs) for comparing reconstructions with
a given reference.
"""

import numpy as np
import odl
import scipy.ndimage.morphology as scimorph

__all__ = ('mean_square_error', 'mean_absolute_error', 'mean_density_value', 
           'density_standard_deviation', 'density_range', 'blurring',
           'false_structures')

def mean_square_error(reco, orig, mask=None):
    """Implementation of mean square error FOM.

    Evaluates mean square error FOM between reconstruction (reco) and original
    data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l2_normSquared(diff) / (2 * (l2_normSquared(reco) +
                                        l2_normSquared(orig)))
    return fom


def mean_absolute_error(reco, orig, mask=None):
    """Implementation of mean absolute error FOM.

    Evaluates mean absolute error FOM between reconstruction (reco) and
    original data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    l1_norm = odl.solvers.L1Norm(reco.space)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l1_norm(diff) / (l1_norm(reco) + l1_norm(orig))
    return fom


def mean_density_value(reco, orig, mask=None):
    """Implementation of mean density value FOM.

    Evaluates mean density value FOM between reconstruction (reco) and original
    data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    reco = reco*mask
    orig = orig*mask
    reco_mean = np.sum(reco)/np.sum(mask)
    orig_mean = np.sum(orig)/np.sum(mask)
    fom = 1 - 0.5 * (np.abs(reco_mean - orig_mean) /
                     (np.abs(reco_mean) + np.abs(orig_mean)))
    return fom


def density_standard_deviation(reco, orig, mask=None):
    """Implementation of density standard deviation FOM.

    Evaluates density standard deviation FOM between reconstruction (reco) and
    original data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    reco = reco*mask
    orig = orig*mask
    # TODO: Make this continuous
    indices = np.where(mask is True)
    reco_std = np.std(reco.asarray()[indices])
    orig_std = np.std(orig.asarray()[indices])
    fom = 1 - (reco_std - orig_std)/(reco_std + orig_std)
    return fom


def density_range(reco, orig, mask=None):
    """Implementation of density range FOM.

    Evaluates density range FOM between reconstruction (reco) and original
    data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape,dtype=bool)
    indices = np.where(mask == True)
    reco_range = np.max(reco.asarray()[indices]) - np.min(reco.asarray()[indices])
    orig_range = np.max(orig.asarray()[indices]) - np.min(orig.asarray()[indices])
    fom = 1 - np.abs(reco_range - orig_range)/np.abs(reco_range + orig_range)
    return fom


def blurring(reco, orig, mask=None):
    """Implementation of blurring FOM.

    Evaluates blurring FOM between reconstruction (reco) and original
    data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    mask = scimorph.distance_transform_edt(1-mask)
    mask = np.exp(-mask/30)
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l2_normSquared(diff)/(2 * (l2_normSquared(reco) +
                                      l2_normSquared(orig)))
    return fom


def false_structures(reco, orig, mask=None):
    """Implementation of false structures FOM.

    Evaluates false structures FOM between reconstruction (reco) and original
    data (orig). Allows for a masking of the two spaces (mask)

    Parameters
    ----------
    reco : `FnBase` or `ProductSpace`
        Input data or reconstruction.
    orig : `FnBase` or `ProductSpace`
        Reference data to compare 'reco' to.
    mask : `FnBase` or `ProductSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    if mask is None:
        mask = np.ones(reco.shape, dtype=bool)
    l2_normSquared = odl.solvers.L2NormSquared(reco.space)
    mask = scimorph.distance_transform_edt(1-mask)
    mask = np.exp(-mask/30)
    if len(np.unique(mask)) != 1:
        mask = 1-mask
    reco = reco*mask
    orig = orig*mask
    norm_factor = 1.0
    diff = (reco - orig)/norm_factor
    fom = 1.0 - l2_normSquared(diff)/(2 * (l2_normSquared(reco) +
                                      l2_normSquared(orig)))
    return fom
