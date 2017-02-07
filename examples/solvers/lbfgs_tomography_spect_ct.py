"""Tomography using the `bfgs_method` solver.

Solves an approximation of the optimization problem

    min_x ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.

The problem is approximated by applying the Moreau envelope to ``|| . ||_1``
which gives a differentiable functional. This functional is equal to the so
called Huber functional.
"""

import numpy as np
import odl


# --- Set up the forward operator (ray transform) --- #


# Discrete reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 200 samples per dimension.
space = odl.uniform_discr([-20, -20], [20, 20], [512, 512])
domain = odl.ProductSpace(space, 2)

geo = odl.tomo.parallel_beam_geometry(space)
rt = odl.tomo.RayTransform(space, geo)

# Create the (nonlinear) forward operator per channel
ct_op = rt * odl.ComponentProjection(domain, 0)
exprt = odl.ufunc_ops.exp(rt.range) * (-0.1 * rt)
pet_op = odl.OperatorPointwiseProduct(
    exprt * odl.ComponentProjection(domain, 0),
    rt * odl.ComponentProjection(domain, 1))

# Combine to create the full forward operator
prod_op = odl.BroadcastOperator(ct_op, pet_op)

# Create phantom
phantom = domain.element([odl.phantom.circle(space, radius=0.95),
                          odl.phantom.derenzo_sources(space)])
phantom.show('phantom')

# Create data
ct_data = ct_op(phantom)
ct_data += odl.phantom.white_noise(ct_data.space) * np.mean(ct_data) * 0.02
ct_data.show('ct_data')

pet_data = pet_op(phantom)
pet_data += odl.phantom.white_noise(pet_data.space) * np.mean(pet_data) * 0.2
pet_data.show('pet_data')

# Create data discrepancy terms
ct_data_discrepancy = odl.solvers.L2NormSquared(ct_data.space).translated(ct_data) * ct_op
pet_data_discrepancy = odl.solvers.L2NormSquared(pet_data.space).translated(pet_data) * pet_op
data_discrepancy = ct_data_discrepancy + pet_data_discrepancy

# Create regularizer
gradient = odl.Gradient(space)
l1_norm = odl.solvers.GroupL1Norm(gradient.range)
smoothed_l1 = odl.solvers.MoreauEnvelope(l1_norm, sigma=0.003)
channel_regularizer = smoothed_l1 * gradient

regularizer = odl.solvers.SeparableSum(0.003 * channel_regularizer,
                                       0.000001 * channel_regularizer)

# Create objective functional
func = data_discrepancy + regularizer

callback = odl.solvers.CallbackShow()

line_search = odl.solvers.BacktrackingLineSearch(data_discrepancy, estimate_step=True)

x = domain.zero()
odl.solvers.bfgs_method(func, x, line_search=line_search,
                        callback=callback)

# Display images
phantom.show(title='original image')
x.show(title='reconstructed image', force_show=True)
