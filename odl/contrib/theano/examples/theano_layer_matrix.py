"""Example of how to convert an ODL operator to a Theano operator layer.

In this example we take an ODL operator given by a `MatrixOperator` and
convert it into a theano operator that can be used inside any theano
computational graph.

We also demonstrate that we can compute the gradient of the scalar-valued
squared L2-norm function properly using either Theano or ODL.
"""

import theano
import theano.tensor as T
import numpy as np
import odl
from odl.contrib import theano as theano_wrapper

# --- Wrap ODL operator as Theano operator --- #

# Define ODL operator
matrix = np.array([[1., 2.],
                   [0., 0.],
                   [0., 1.]])
odl_op = odl.MatrixOperator(matrix)

# Define evaluation point
x = [1., 2.]

# Create theano placeholders
x_theano = T.fvector('x')

# Create theano layer from ODL operator
odl_op_layer = theano_wrapper.TheanoOperator(odl_op)

# Build computation graph
y_theano = odl_op_layer(x_theano)
y_theano_func = theano.function([x_theano], y_theano)

# Evaluate using theano and compare to odl_op(x)
print('Theano eval    : ', y_theano_func(x))
print('ODL eval       : ', odl_op(x))

# --- Wrap ODL functional as Theano operator --- #

# Define ODL cost and composed functional
odl_cost = odl.solvers.L2NormSquared(odl_op.range)
odl_functional = odl_cost * odl_op

# Create theano layer from ODL cost
cost_theano_layer = theano_wrapper.TheanoOperator(odl_cost)

# Build computation graph for the gradient of the composed cost wrt x
y_theano = odl_op_layer(x_theano)
cost_theano = cost_theano_layer(y_theano)
cost_grad_theano = theano.grad(cost_theano, x_theano)
cost_theano_grad_func = theano.function([x_theano], cost_grad_theano)

# Compute gradient at x and compare to ODL functional.gradient(x)
print('Theano gradient: ', cost_theano_grad_func(x))
print('ODL gradient   : ', odl_functional.gradient(x))
