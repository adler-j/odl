# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL spaces to tensorflow layers."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

import tensorflow as tf
import numpy as np
import odl


__all__ = ('TensorflowOperator',)


class TensorflowOperator(odl.Operator):
    def __init__(self, input_tensor, output_tensor, linear=False, sess=None):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

        # TODO: Fix with tensors
        domain = odl.fn(np.prod(input_tensor.shape.as_list()),
                        dtype=input_tensor.dtype.as_numpy_dtype)
        range = odl.fn(np.prod(output_tensor.shape.as_list()),
                       dtype=output_tensor.dtype.as_numpy_dtype)

        self.__adjoint_of_derivative_tensor = None
        self.__derivative_tensor = None

        if sess is None:
            self.sess = tf.get_default_session()
        else:
            self.sess = sess

        super(TensorflowOperator, self).__init__(domain, range, linear=linear)

    @property
    def adjoint_of_derivative_tensor(self):
        """Adjoint of the derivative of the output w.r.t. the input.

        Implemented as a property to allow lazy evaluation. This speeds up
        graph creation and defers errors to when the method is called.
        """
        if self.__adjoint_of_derivative_tensor is None:
            self.dy = tf.placeholder(self.output_tensor.dtype,
                                     shape=self.output_tensor.shape)
            adjoint_of_derivative_tensor = tf.gradients(
                self.output_tensor, [self.input_tensor], [self.dy])[0]
            self.adjoint_of_derivative_tensor = (range.weighting.const *
                                                 adjoint_of_derivative_tensor)
        return self.__adjoint_of_derivative_tensor

    @property
    def derivative_tensor(self):
        """Derivative of the output w.r.t. the input.

        Implemented as a property to allow lazy evaluation. This speeds up
        graph creation and defers errors to when the method is called.
        """
        if self.__derivative_tensor is None:
            # Since tensorflow does not support forward differentiation, use
            # trick that adjoint of the derivative of adjoint of the derivative
            # is simply the derivative.
            self.dx = tf.placeholder(self.input_tensor.dtype,
                                     shape=self.input_tensor.shape)
            derivative_tensor = tf.gradients(
                self.adjoint_of_derivative_tensor, [self.dy], [self.dx])[0]
            self.__derivative_tensor = derivative_tensor
        return self.__derivative_tensor

    def _call(self, x):
        x_reshaped = np.reshape(np.asarray(x), self.input_tensor.shape)

        result = self.sess.run(self.output_tensor,
                               feed_dict={self.input_tensor: x_reshaped})

        return np.ravel(result)

    def derivative(self, x):
        op = self

        class TensorflowOperatorDerivative(odl.Operator):
            def _call(self, dx):
                result = op.sess.run(op.derivative_tensor,
                                     feed_dict={op.input_tensor: np.asarray(x),
                                                op.dx: np.asarray(dx)})

                return result

            @property
            def adjoint(self):
                class TensorflowOperatorDerivativeAdjoint(odl.Operator):
                    def _call(self, y):
                        result = op.sess.run(
                            op.adjoint_of_derivative_tensor,
                            feed_dict={op.input_tensor: np.asarray(x),
                                       op.dy: np.asarray(y)})

                        return result

                return TensorflowOperatorDerivativeAdjoint(self.range,
                                                           self.domain,
                                                           linear=True)

        return TensorflowOperatorDerivative(self.domain,
                                            self.range,
                                            linear=True)


if __name__ == '__main__':
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
