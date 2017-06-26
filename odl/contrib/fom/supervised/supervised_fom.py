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

__all__ = ('mean_square_error', 'mean_absolute_error', 'mean_value_difference',
           'standard_deviation_difference', 'range_difference', 'blurring',
           'false_structures')


def mean_square_error(test_data, ref_data, mask=None, normalized=False):
    """Mean square error-based FOM between ``test_data`` and ``ref_data``.

    Evaluates mean square error between input (``test_data``) and
    reference data (``ref_data``), by measuring consistency using the L2 -norm.
    Allows for normalization (``normalized``) and a masking of the two spaces
    (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \| f - g \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\| f - g \|^2_2}{\| f \|^2_2 + \| g \|^2_2}.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElementElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating mean square error between ``test_data`` and
        ``ref_data``. In normalized form the FOM takes values in [0, 1], with
        higher correspondance at lower FOM value.
    """

    import odl

    l2_normSquared = odl.solvers.L2NormSquared(test_data.space)

    if mask is not None:
        test_data = test_data * mask
        ref_data = ref_data * mask

    diff = test_data - ref_data
    fom = l2_normSquared(diff)

    if normalized:
            fom /= (l2_normSquared(test_data) + l2_normSquared(ref_data))

    return fom


def mean_absolute_error(test_data, ref_data, mask=None, normalized=False):
    """Mean absolute error-based FOM between ``test_data`` and ``ref_data``.

    Evaluates mean absolute error between input (``test_data``) and
    reference data (``ref_data``), by measuring consistency using the L1-norm.
    Allows for normalization (``normalized``) and a masking of the two spaces
    (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \| f - g \|_1,

    or, in normalized form

    .. math::
        \\frac{\| f - g \|_1}{\| f \|_1 + \| g \|_1}.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating mean absolute error between ``test_data`` and
        ``ref_data``. In normalized form the FOM takes values in [0, 1], with
        higher correspondance at lower FOM value.
    """

    import odl

    l1_norm = odl.solvers.L1Norm(test_data.space)
    if mask:
        test_data = test_data * mask
        ref_data = ref_data * mask
    diff = test_data - ref_data
    fom = l1_norm(diff)

    if normalized:
        fom /= (l1_norm(test_data) + l1_norm(ref_data))

    return fom


def mean_value_difference(test_data, ref_data, mask=None, normalized=False):
    """Mean value-based FOM between ``test_data`` and ``ref_data``.

    Evaluates difference in mean value between input (``test_data``) and
    reference data (``ref_data``). Allows for normalization (``normalized``)
    and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
         \\lvert  \\overline{f} - \\overline{g} \\rvert,

    or, in normalized form

    .. math::
        \\bigg \\lvert \\frac{ \\overline{f} - \\overline{g} }
                                 {\\overline{f} + \\overline{g}} \\bigg
                                                                 \\rvert.

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating difference in mean value between
        ``test_data`` and ``ref_data``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
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

    fom = np.abs((test_data_mean - ref_data_mean))

    if normalized:
        fom /= (np.abs(test_data_mean) + np.abs(ref_data_mean))

    return fom


def standard_deviation_difference(test_data, ref_data, mask=None,
                                  normalized=False):
    """Standard deviation-based FOM between ``test_data`` and ``ref_data``.

    Evaluates difference in standard deviation between input (``test_data``)
    and reference data (``ref_data``). Allows normalization (``normalized``)
    and a masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
         \\lvert \| f - \\overline{f} \|_2 -
                 \| g - \\overline{g} \|_2 \\rvert,

    or, in normalized form

    .. math::
        \\bigg \\lvert \\frac{\| f - \\overline{f} \|_2 -
                              \| g - \\overline{g} \|_2}
                             {\| f - \\overline{f} \|_2 +
                              \| g - \\overline{g} \|_2 } \\bigg \\rvert,

    where

    .. math::
        \\overline{f} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega f dx,

    and

    .. math::
        \\overline{g} := \\frac{1}{\|1_\Omega\|_1} \\int_\Omega g dx.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating absolute difference in standard deviation
        between ``test_data`` and ``ref_data``. In normalized form the FOM
        takes values in [0, 1], with higher correspondance at lower FOM value.
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

    fom = np.abs((l2_norm(test_data - test_data_mean) -
                  l2_norm(ref_data - ref_data_mean)))

    if normalized:
        fom /= (l2_norm(test_data - test_data_mean) +
                l2_norm(ref_data - ref_data_mean))

    return fom


def range_difference(test_data, ref_data, mask=None, normalized=False):
    """Range-based FOM between ``test_data`` and ``ref_data``.

    Evaluates difference in range between input (``test_data``) and reference
    data (``ref_data``). Allows for normalization (``normalized``) and a
    masking of the two spaces (``mask``).

    Notes
    ----------
    The FOM evaluates

    .. math::
        \\lvert \\left(\\max(f) - \\min(f) \\right) -
                \\left(\\max(g) - \\min(g) \\right) \\rvert

    or, in normalized form

    .. math::
        \\bigg \\lvert \\frac{\\left(\\max(f) - \\min(f) \\right) -
                              \\left(\\max(g) - \\min(g)\\right)}
                             {\\left(\\max(f) - \\min(f)\\right) +
                              \\left(\\max(g) - \\min(g)\\right)}
        \\bigg \\rvert

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating absolute difference in range between
        ``test_data`` and ``ref_data``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
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

    fom = np.abs(test_data_range - ref_data_range)

    if normalized:
        fom /= np.abs(test_data_range + ref_data_range)

    return fom


def blurring(test_data, ref_data, mask=None, normalized=False,
             weight_factor=30):

    """Blurring-based FOM between ``test_data`` and ``ref_data``.

    Evaluates mean square error between input (``test_data``) and reference
    data (``ref_data``) using an added binary mask (``mask``), such that the
    error is weighted with higher importance given to the defined
    structure-of-interest. Allows for normalization (``normalized``).

    NOTE: If omitting the mask argument, the blurring FOM is equivalent to the
    mean square error FOM.

    Notes
    ----------
    The FOM evaluates

    .. math::
        \|\\alpha (f - g) \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\| \\alpha (f - g) \|^2_2}{\| \\alpha f \|^2_2 +
                                  \| \\alpha  g \|^2_2},

    where :math:`\\alpha` is a weighting function with higher values near a
    structure of interest defined by ``mask``. The weighting function is given
    as

    .. math::
        \\alpha = e^{-\\frac{1}{k} \\beta},

    where :math:`\\beta(x)` is the Euclidian distance transform from :math:`x`
    to the complement of the structure of interest, and :math:`k` is a positive
    real number.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare ``test_data`` to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.
    weight_factor : `Scalar`
        Positive real number for scaling of weighted mask. Higher value gives
        smoother weighting.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating weighted mean square error between
        ``test_data`` and ``ref_data``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np
    import odl.contrib.fom
    import scipy.ndimage.morphology as scimorph

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(-mask / weight_factor)

    fom = odl.contrib.fom.mean_square_error(test_data,
                                            ref_data,
                                            mask=mask,
                                            normalized=normalized)

    return fom


def false_structures(test_data, ref_data, mask=None, normalized=False,
                     weight_factor=30):

    """False structures-based FOM between ``test_data`` and ``ref_data``.

    Evaluates mean square error between input (``test_data``) and reference
    data (``ref_data``) using an added binary mask (``mask``), such that the
    error is weighted with lower importance given to the defined
    structure-of-interest. Allows for normalization (``normalized``).

    NOTE: If omitting the mask argument, the blurring FOM is equivalent to the
    mean square error FOM.

    Notes
    ----------
    The FOM evaluates

    .. math::
        \\bigg \| \\frac{1}{\\alpha} (f - g) \\bigg \|^2_2,

    or, in normalized form

    .. math::
        \\frac{\\bigg \| \\frac{1}{\\alpha} (f - g) \\bigg \|^2_2}
              {\\bigg \| \\frac{1}{\\alpha} f \\bigg \|^2_2 +
               \\bigg \| \\frac{1}{\\alpha} g \\bigg \|^2_2},

    where :math:`\\alpha` is a weighting function with higher values near a
    structure of interest defined by ``mask``. The weighting function is given
    as

    .. math::
        \\alpha = e^{-\\frac{1}{k} \\beta},

    where :math:`\\beta(x)` is the Euclidian distance transform from :math:`x`
    to the complement of the structure of interest, and :math:`k` is a positive
    real number.

    The normalized FOM takes its values in [0, 1], with higher correspondance
    at lower FOM value.

    Parameters
    ----------
    test_data : `LinearSpaceElement`
        Input data or reconstruction.
    ref_data : `LinearSpaceElement`
        Reference data to compare 'test_data' to.
    mask : `LinearSpaceElement`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : `Boolean`
        Boolean flag to switch between unormalized and normalized FOM.
    weight_factor : `Scalar`
        Positive real number for scaling of weighted mask. Higher value gives
        smoother weighting.

    Returns
    -------
    fom : `Scalar`
        Scalar (float) indicating weighted mean square error between
        ``test_data`` and ``ref_data``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np
    import odl.contrib.fom
    import scipy.ndimage.morphology as scimorph

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(mask / weight_factor)

    fom = odl.contrib.fom.mean_square_error(test_data,
                                            ref_data,
                                            mask=mask,
                                            normalized=normalized)

    return fom
