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

import odl

__all__ = ('mean_squared_error', 'mean_absolute_error',
           'mean_value_difference', 'standard_deviation_difference',
           'range_difference', 'blurring', 'false_structures',
           'ssim')


def mean_squared_error(data, ground_truth, mask=None, normalized=False):
    """FOM returning the L2-distance between ``data`` and ``ground_truth``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between
    input (``data``) and reference (``ground_truth``), by measuring
    consistency using the L2 -norm. Allows for normalization (``normalized``)
    and a masking of the two spaces (``mask``).

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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating mean squared error between ``data`` and
        ``ground_truth``. In normalized form the FOM takes values in
        [0, 1], with higher correspondance at lower FOM value.
    """

    l2_normSquared = odl.solvers.L2NormSquared(data.space)

    if mask is not None:
        data = data * mask
        ground_truth = ground_truth * mask

    diff = data - ground_truth
    fom = l2_normSquared(diff)

    if normalized:
            fom /= (l2_normSquared(data) + l2_normSquared(ground_truth))

    return fom


def mean_absolute_error(data, ground_truth, mask=None, normalized=False):
    """FOM returning the L1-distance between ``data`` and ``ground_truth``.

    Evaluates `mean absolute error
    <https://en.wikipedia.org/wiki/Mean_absolute_error>`_ between
    input (``data``) and reference (``ground_truth``), by measuring
    consistency using the L1-norm. Allows for normalization (``normalized``)
    and a masking of the two spaces (``mask``).

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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating mean absolute error between ``data`` and
        ``ground_truth``. In normalized form the FOM takes values in
        [0, 1], with higher correspondance at lower FOM value.
    """

    l1_norm = odl.solvers.L1Norm(data.space)
    if mask:
        data = data * mask
        ground_truth = ground_truth * mask
    diff = data - ground_truth
    fom = l1_norm(diff)

    if normalized:
        fom /= (l1_norm(data) + l1_norm(ground_truth))

    return fom


def mean_value_difference(data, ground_truth, mask=None, normalized=False):
    """FOM returning the  difference in mean value between ``data``
    and ``ground_truth``.

    Evaluates difference in `mean value
    <https://en.wikipedia.org/wiki/Mean_of_a_function>`_ between input
    (``data``) and reference (``ground_truth``). Allows for normalization
    (``normalized``) and a masking of the two spaces (``mask``).

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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating difference in mean value between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np

    l1_norm = odl.solvers.L1Norm(data.space)

    if mask:
        data = data * mask
        ground_truth = ground_truth * mask

    # Identify positive part of data
    pos_data = data.copy()
    pos_data = pos_data.asarray()
    pos_data[pos_data < 0] = 0
    pos_data = data.space.element(pos_data)

    # Identify negative part of data
    neg_data = data.copy()
    neg_data = neg_data.asarray()
    neg_data[neg_data > 0] = 0
    neg_data = data.space.element(neg_data)

    # Identify positive part of ground_truth
    pos_ground_truth = ground_truth.copy()
    pos_ground_truth = pos_ground_truth.asarray()
    pos_ground_truth[pos_ground_truth < 0] = 0
    pos_ground_truth = ground_truth.space.element(pos_ground_truth)

    # Identify negative part of ground_truth
    neg_ground_truth = ground_truth.copy()
    neg_ground_truth = neg_ground_truth.asarray()
    neg_ground_truth[neg_ground_truth > 0] = 0
    neg_ground_truth = ground_truth.space.element(neg_ground_truth)

    # Volume of space
    vol = data.space.domain.volume

    # Mean value of data
    data_mean = (1 / vol) * (l1_norm(pos_data) -
                             l1_norm(neg_data))

    # Mean value of ground_truth
    ground_truth_mean = (1 / vol) * (l1_norm(pos_ground_truth) -
                                     l1_norm(neg_ground_truth))

    fom = np.abs((data_mean - ground_truth_mean))

    if normalized:
        fom /= (np.abs(data_mean) + np.abs(ground_truth_mean))

    return fom


def standard_deviation_difference(data, ground_truth, mask=None,
                                  normalized=False):
    """FOM returning absolute difference in standard deviation between
    ``data`` and ``ground_truth``.

    Evaluates difference in standard deviation between input (``data``)
    and reference (``ground_truth``). Allows normalization (``normalized``)
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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating absolute difference in standard deviation
        between ``data`` and ``ground_truth``. In normalized form the FOM
        takes values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np

    l1_norm = odl.solvers.L1Norm(data.space)
    l2_norm = odl.solvers.L2Norm(data.space)

    if mask:
        data = data * mask
        ground_truth = ground_truth * mask

    # Identify positive part of data
    pos_data = data.copy()
    pos_data = pos_data.asarray()
    pos_data[pos_data < 0] = 0
    pos_data = data.space.element(pos_data)

    # Identify negative part of data
    neg_data = data.copy()
    neg_data = neg_data.asarray()
    neg_data[neg_data > 0] = 0
    neg_data = data.space.element(neg_data)

    # Identify positive part of ground_truth
    pos_ground_truth = ground_truth.copy()
    pos_ground_truth = pos_ground_truth.asarray()
    pos_ground_truth[pos_ground_truth < 0] = 0
    pos_ground_truth = ground_truth.space.element(pos_ground_truth)

    # Identify negative part of ground_truth
    neg_ground_truth = ground_truth.copy()
    neg_ground_truth = neg_ground_truth.asarray()
    neg_ground_truth[neg_ground_truth > 0] = 0
    neg_ground_truth = ground_truth.space.element(neg_ground_truth)

    # Volume of space
    vol = data.space.domain.volume

    # Mean value of data
    data_mean = (1 / vol) * (l1_norm(pos_data) -
                             l1_norm(neg_data))

    # Mean value of ground_truth
    ground_truth_mean = (1 / vol) * (l1_norm(pos_ground_truth) -
                                     l1_norm(neg_ground_truth))

    fom = np.abs((l2_norm(data - data_mean) -
                  l2_norm(ground_truth - ground_truth_mean)))

    if normalized:
        fom /= (l2_norm(data - data_mean) +
                l2_norm(ground_truth - ground_truth_mean))

    return fom


def range_difference(data, ground_truth, mask=None, normalized=False):
    """FOM returning difference in range between ``data`` and ``ground_truth``.

    Evaluates difference in range between input (``data``) and reference
    data (``ground_truth``). Allows for normalization (``normalized``) and a
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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.

    Returns
    -------
    fom : float
        Scalar (float) indicating absolute difference in range between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np

    if mask:
        indices = np.where(mask is True)
        data_range = (np.max(data.asarray()[indices]) -
                      np.min(data.asarray()[indices]))
        ground_truth_range = (np.max(ground_truth.asarray()[indices]) -
                              np.min(ground_truth.asarray()[indices]))
    else:
        data_range = np.max(data) - np.min(data)
        ground_truth_range = np.max(ground_truth) - np.min(ground_truth)

    fom = np.abs(data_range - ground_truth_range)

    if normalized:
        fom /= np.abs(data_range + ground_truth_range)

    return fom


def blurring(data, ground_truth, mask=None, normalized=False,
             weight_factor=30):
    """FOM returning weighted L2-distance between ``data`` and ``ground_truth``,
        emphasizing regions defined by ``mask``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between input
    (``data``) and reference data (``ground_truth``) using an added binary
    mask (``mask``), such that the error is weighted with higher importance
    given to the defined structure-of-interest. Allows for normalization
    (``normalized``).

    NOTE: If omitting the mask argument, the blurring FOM is equivalent to the
    mean squared error FOM.

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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.
    weight_factor : float
        Positive real number for scaling of weighted mask. Higher value gives
        smoother weighting.

    Returns
    -------
    fom : float
        Scalar (float) indicating weighted mean squared error between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np
    import odl.contrib.fom
    import scipy.ndimage.morphology as scimorph

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(-mask / weight_factor)

    fom = odl.contrib.fom.mean_squared_error(data,
                                             ground_truth,
                                             mask=mask,
                                             normalized=normalized)

    return fom


def false_structures(data, ground_truth, mask=None, normalized=False,
                     weight_factor=30):
    """FOM returning weighted L2-distance between ``data`` and ``ground_truth``,
        emphasizing complement of regions defined by ``mask``.

    Evaluates `mean squared error
    <https://en.wikipedia.org/wiki/Mean_squared_error>`_ between input
    (``data``) and reference data (``ground_truth``) using an added binary
    mask (``mask``), such that the error is weighted with lower importance
    given to the defined structure-of-interest. Allows for normalization
    (``normalized``).

    NOTE: If omitting the mask argument, the blurring FOM is equivalent to the
    mean squared error FOM.

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
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare 'data' to.
    mask : `FnBaseVector`
        Binary mask to define ROI in which FOM evaluation is performed.
    normalized  : bool
        Boolean flag to switch between unormalized and normalized FOM.
    weight_factor : float
        Positive real number for scaling of weighted mask. Higher value gives
        smoother weighting.

    Returns
    -------
    fom : float
        Scalar (float) indicating weighted mean squared error between
        ``data`` and ``ground_truth``. In normalized form the FOM takes
        values in [0, 1], with higher correspondance at lower FOM value.
    """

    import numpy as np
    import odl.contrib.fom
    import scipy.ndimage.morphology as scimorph

    if mask is not None:
        mask = scimorph.distance_transform_edt(1 - mask)
        mask = np.exp(mask / weight_factor)

    fom = odl.contrib.fom.mean_squared_error(data,
                                             ground_truth,
                                             mask=mask,
                                             normalized=normalized)

    return fom


def ssim(data, ground_truth,
         size=11, sigma=1.5, K1=0.01, K2=0.03, dynamic_range=None,
         normalized=False):
    """Structural SIMilarity between ``data`` and ``ground_truth``.

    Evaluates `structural similarity
    <https://en.wikipedia.org/wiki/Structural_similarity>`_ between
    input (``data``) and reference (``ground_truth``).

    Parameters
    ----------
    data : `FnBaseVector`
        Input data or reconstruction.
    ground_truth : `FnBaseVector`
        Reference to compare ``data`` to.
    normalized : bool
        TEXT

    Returns
    -------
    fom : float
        Scalar (float) indicating structural similarity between ``data`` and
        ``ground_truth``. Takes values in [-1, 1], with -1 indicating full
        dis-similarity and 1 full similarity. Uncorrelated images will have
        similarity index 0. In normalized form the FOM values are rescaled to
        [0, 1], with higher correspondance at lower FOM value.
    """
    from scipy.signal import fftconvolve
    import numpy as np

    data = np.asarray(data)
    ground_truth = np.asarray(ground_truth)
    ndim = data.ndim

    # Compute gaussian
    coords = np.meshgrid(*(ndim * (np.linspace(-(size - 1)/2,
                                               (size - 1)/2, size),)))
    window = np.exp(-(sum(xi**2 for xi in coords) / (2.0 * sigma**2)))
    window /= np.sum(window)

    def smoothen(img):
        return fftconvolve(window, img, mode='valid')

    if dynamic_range is None:
        dynamic_range = np.max(ground_truth) - np.min(ground_truth)

    C1 = (K1 * dynamic_range)**2
    C2 = (K2 * dynamic_range)**2
    mu1 = smoothen(data)
    mu2 = smoothen(ground_truth)
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = smoothen(data * data) - mu1_sq
    sigma2_sq = smoothen(ground_truth * ground_truth) - mu2_sq
    sigma12 = smoothen(data * ground_truth) - mu1_mu2

    nom = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    pointwise_ssim = nom/denom

    ssim = np.mean(pointwise_ssim)

    if normalized:
        return 0.5 - ssim / 2
    else:
        return ssim
