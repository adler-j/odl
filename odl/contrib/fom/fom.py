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

__all__ = ('mean_square_error', 'mean_absolute_error', 'mean_density_value',
           'density_standard_deviation', 'density_range', 'blurring',
           'false_structures')


def mean_square_error(test_data, ref_data, mask=None):
    """Mean square error between test_data and ref_data.

    Evaluates mean square error FOM between reconstruction (test_data) and
    reference data (ref_data), by measuring consistency using the L 2 -norm.
    Allows for a masking of the two spaces (mask).

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'reco' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluation is performed.

    Returns
    -------
    fom : scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """

    import odl

    l2_normSquared = odl.solvers.L2NormSquared(test_data.space)
    norm_factor = 1.0
    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask
    diff = (test_data - ref_data) / norm_factor
    fom = 1.0 - l2_normSquared(diff) / (2 * (l2_normSquared(test_data) +
                                        l2_normSquared(ref_data)))
    return fom


def mean_absolute_error(test_data, ref_data, mask=None):
    """Mean absolute error FOM.

    Evaluates mean absolute error FOM between reconstruction (test_data) and
    reference data (ref_data). Allows for a masking of the two spaces (mask).

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'reco' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """

    import odl

    l1_norm = odl.solvers.L1Norm(test_data.space)
    norm_factor = 1.0
    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask
    diff = (test_data - ref_data) / norm_factor
    fom = 1.0 - l1_norm(diff) / (l1_norm(test_data) + l1_norm(ref_data))
    return fom


def mean_density_value(test_data, ref_data, mask=None):
    """Mean density value FOM.

    Evaluates mean density value FOM between reconstruction (test_data)
    and reference data (ref_data). Allows for a masking of the two spaces
    (mask).

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'reco' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """

    import numpy as np

    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask
        test_data_mean = np.sum(test_data) / np.sum(mask)
        ref_data_mean = np.sum(ref_data) / np.sum(mask)
    else:
        test_data_mean = np.mean(test_data)
        ref_data_mean = np.mean(ref_data)

    fom = 1 - 0.5 * (np.abs(test_data_mean - ref_data_mean) /
                     (np.abs(test_data_mean) + np.abs(ref_data_mean)))
    return fom


def density_standard_deviation(test_data, ref_data, mask=None):
    """Density standard deviation FOM.

    Evaluates density standard deviation FOM between reconstruction
    (test_data) and reference data (ref_data). Allows for a masking of the
    two spaces (mask).reco

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'test_data' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    import numpy as np

    # TODO: Make this continuous
    if mask is not None:
        test_data = test_data * mask
        ref_data = ref_data * mask
        indices = np.where(mask is ref_data)  # TODO: Fix (mask == True)?
        print(indices)
        test_data_std = np.std(test_data.asarray()[indices])
        ref_data_std = np.std(ref_data.asarray()[indices])
    else:
        test_data_std = np.std(test_data)
        ref_data_std = np.std(ref_data)

    fom = 1 - (test_data_std - ref_data_std) / (test_data_std + ref_data_std)
    return fom


def density_range(test_data, ref_data, mask=None):
    """Density range FOM.

    Evaluates density range FOM between reconstruction (test_data) and
    reference data (ref_data). Allows for a masking of the two spaces (mask).

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'test_data' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    import numpy as np

    if mask:
        indices = np.where(mask is True)
        test_data_range = (np.max(test_data.asarray()[indices]) -
                           np.min(test_data.asarray()[indices]))
        ref_data_range = (np.max(ref_data.asarray()[indices]) -
                          np.min(ref_data.asarray()[indices]))
    else:
        test_data_range = np.max(test_data) - np.min(test_data)
        ref_data_range = np.max(ref_data) - np.min(ref_data)

    fom = 1 - (np.abs(test_data_range - ref_data_range) /
               np.abs(test_data_range + ref_data_range))
    return fom


def blurring(test_data, ref_data, mask=None):
    """Blurring FOM.

    Evaluates blurring FOM between reconstruction (test_data) and reference
    data (ref_data). Allows for a masking of the two spaces (mask). NOTE:
    without mask the blurring FOM is equivalent to the mean square error FOM.

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'test_data' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    import numpy as np
    import odl
    import scipy.ndimage.morphology as scimorph

    l2_normSquared = odl.solvers.L2NormSquared(test_data.space)
    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(-mask / 30)
        test_data = test_data * mask
        ref_data = ref_data * mask
    norm_factor = 1.0
    diff = (test_data - ref_data) / norm_factor
    fom = 1.0 - l2_normSquared(diff) / (2 * (l2_normSquared(test_data) +
                                        l2_normSquared(ref_data)))
    return fom


def false_structures(test_data, ref_data, mask=None):
    """False structures FOM.

    Evaluates false structures FOM between reconstruction (test_data) and
    reference data (ref_data). Allows for a masking of the two spaces (mask).

    Parameters
    ----------
    test_data : `LinearSpace`
        Input data or reconstruction.
    ref_data : `LinearSpace`
        Reference data to compare 'test_data' to.
    mask : `LinearSpace`
        Binary mask to define ROI in which FOM evaluatoin is performed.

    Returns
    -------
    fom: scalar
        Scalar (float) indicating perfect correspondance (1) to no
        correspondance (0)
    """
    import numpy as np
    import odl
    import scipy.ndimage.morphology as scimorph

    l2_normSquared = odl.solvers.L2NormSquared(test_data.space)
    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(-mask / 30)
        if len(np.unique(mask)) != 1:
            mask = 1 - mask
        test_data = test_data * mask
        ref_data = ref_data * mask
    norm_factor = 1.0
    diff = (test_data - ref_data) / norm_factor
    fom = 1.0 - l2_normSquared(diff) / (2 * (l2_normSquared(test_data) +
                                        l2_normSquared(ref_data)))
    return fom
