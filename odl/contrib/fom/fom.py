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
    """Mean square error-based FOM between test_data and ref_data.

    Evaluates mean square error FOM between input (test_data) and
    reference data (ref_data), by measuring consistency using the L2 -norm.
    Allows for a masking of the two spaces (mask).

    Notes
    ----------
    The FOM evaluates

    .. math::
        1 - \\frac{\| f - g \|^2_2}{\| f \|^2_2 + \| g \|^2_2}.

    The FOM takes its values in [0 , 1], with higher correspondance
    at higher FOM value.

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

    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask

    diff = test_data - ref_data
    fom = 1.0 - l2_normSquared(diff) / (l2_normSquared(test_data) +
                                        l2_normSquared(ref_data))
    return fom


def mean_absolute_error(test_data, ref_data, mask=None):
    """Mean absolute error-based FOM between test_data and ref_data.

    Evaluates mean absolute error FOM between input (test_data) and
    reference data (ref_data), by measuring consistency using the L1-norm.
    Allows for a masking of the two spaces (mask).

    Notes
    ----------
    The FOM evaluates

    .. math::
        1 - \\frac{\| f - g \|_1}{\| f \|_1 + \| g \|_1}.

    The FOM takes its values in [0 , 1], with higher correspondance
    at higher FOM value.

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
    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask
    diff = test_data - ref_data
    fom = 1.0 - l1_norm(diff) / (l1_norm(test_data) + l1_norm(ref_data))
    return fom


def mean_density_value(test_data, ref_data, mask=None):
    """Mean density value-based FOM betweent test_data and ref_data.

    Evaluates mean density value FOM between input (test_data) and
    reference data (ref_data). Allows for a masking of the two spaces (mask).

    Notes
    ----------
    The FOM evaluates

    .. math::
        1 - \\bigg \\lvert \\frac{ \\overline{f} - \\overline{g} }
                                 {\\overline{f} + \\overline{g}} \\bigg
                                                                 \\rvert.

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The FOM takes its values in [0 , 1], with higher correspondance
    at higher FOM value.

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
    import odl

    l1_norm = odl.solvers.L1Norm(test_data.space)

    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask

    # Identify positive part of test_data
    pos_test_data = test_data.copy()
    pos_test_data = pos_test_data.asarray()
    pos_test_data[pos_test_data < 0] = 0
    pos_test_data = test_data.space.element(pos_test_data)

    # Identify negative part of test_data
    neg_test_data = test_data.copy()
    neg_test_data = neg_test_data.asarray()
    neg_test_data[neg_test_data > 0] = 0
    neg_test_data = test_data.space.element(neg_test_data)

    # Identify positive part of ref_data
    pos_ref_data = ref_data.copy()
    pos_ref_data = pos_ref_data.asarray()
    pos_ref_data[pos_ref_data < 0] = 0
    pos_ref_data = ref_data.space.element(pos_ref_data)

    # Identify negative part of ref_data
    neg_ref_data = ref_data.copy()
    neg_ref_data = neg_ref_data.asarray()
    neg_ref_data[neg_ref_data > 0] = 0
    neg_ref_data = ref_data.space.element(neg_ref_data)

    # Volume of space
    vol = l1_norm(test_data.space.one())

    # Mean value of test_data
    test_data_mean = (1 / vol) * (l1_norm(pos_test_data) -
                                  l1_norm(neg_test_data))

    # Mean value of ref_data
    ref_data_mean = (1 / vol) * (l1_norm(pos_ref_data) -
                                 l1_norm(neg_ref_data))

    fom = 1 - (np.abs((test_data_mean - ref_data_mean)) /
               (np.abs(test_data_mean) + np.abs(ref_data_mean)))

    return fom


def density_standard_deviation(test_data, ref_data, mask=None):
    """Density standard deviation-based FOM between test_data and ref_data.

    Evaluates density standard deviation FOM between input (test_data) and
    reference data (ref_data). Allows for a masking of the two spaces (mask).

    Notes
    ----------
    The FOM evaluates

    .. math::
        1 - \\bigg \\lvert \\frac{\| f - \\overline{f} \|_2 -
                                  \| g - \\overline{g} \|_2}
                                 {\| f - \\overline{f} \|_2 +
                                  \| g - \\overline{g} \|_2 } \\bigg \\rvert,

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The FOM takes its values in [0 , 1], with higher correspondance
    at higher FOM value.

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

    l1_norm = odl.solvers.L1Norm(test_data.space)
    l2_norm = odl.solvers.L2Norm(test_data.space)

    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask

    # Identify positive part of test_data
    pos_test_data = test_data.copy()
    pos_test_data = pos_test_data.asarray()
    pos_test_data[pos_test_data < 0] = 0
    pos_test_data = test_data.space.element(pos_test_data)

    # Identify negative part of test_data
    neg_test_data = test_data.copy()
    neg_test_data = neg_test_data.asarray()
    neg_test_data[neg_test_data > 0] = 0
    neg_test_data = test_data.space.element(neg_test_data)

    # Identify positive part of ref_data
    pos_ref_data = ref_data.copy()
    pos_ref_data = pos_ref_data.asarray()
    pos_ref_data[pos_ref_data < 0] = 0
    pos_ref_data = ref_data.space.element(pos_ref_data)

    # Identify negative part of ref_data
    neg_ref_data = ref_data.copy()
    neg_ref_data = neg_ref_data.asarray()
    neg_ref_data[neg_ref_data > 0] = 0
    neg_ref_data = ref_data.space.element(neg_ref_data)

    # Volume of space
    vol = l1_norm(test_data.space.one())

    # Mean value of test_data
    test_data_mean = (1 / vol) * (l1_norm(pos_test_data) -
                                  l1_norm(neg_test_data))

    # Mean value of ref_data
    ref_data_mean = (1 / vol) * (l1_norm(pos_ref_data) -
                                 l1_norm(neg_ref_data))

    fom = 1 - np.abs((l2_norm(test_data - test_data_mean) -
                      l2_norm(ref_data - ref_data_mean)) /
                     ((l2_norm(test_data - test_data_mean) +
                       l2_norm(ref_data - ref_data_mean))))

    return fom
mdv = []


def density_range(test_data, ref_data, mask=None):
    """Density range-based FOM between test_data and ref_data.

    Evaluates density range FOM between input (test_data) and reference
    data (ref_data). Allows for a masking of the two spaces (mask).

    Notes
    ----------
    The FOM evaluates

    .. math::
        1 - \\bigg \\lvert \\frac{\\left(\\max(f) - \\min(f) \\right) -
                                  \\left(\\max(g) - \\min(g)\\right)}
                                 {\\left(\\max(f) - \\min(f)\\right) +
                                  \\left(\\max(g) - \\min(g)\\right)}
            \\bigg \\rvert

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

    Notes
    ----------
    DOCUMENTATION IN PROGRESS

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

    Notes
    ----------
    DOCUMENTATION IN PROGRESS

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
