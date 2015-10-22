# Copyright 2014, 2015 The ODL development group
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


# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import

from future import standard_library
standard_library.install_aliases()
from builtins import super

# External module imports
import pytest
import numpy as np

# ODL imports
import odl
import odl.operator.solvers as solvers
from odl.util.testutils import all_almost_equal


class MultiplyOp(odl.Operator):
    """Multiply with a matrix."""

    def __init__(self, matrix, domain=None, range=None):
        dom = odl.Rn(matrix.shape[1]) if domain is None else domain
        ran = odl.Rn(matrix.shape[0]) if range is None else range
        super().__init__(dom, ran, linear=True)
        self.matrix = matrix

    def _apply(self, rhs, out):
        out.data[:] = np.dot(self.matrix, rhs.data)

    @property
    def domain(self):
        return self._domain

    @property
    def range(self):
        return self._range

    @property
    def adjoint(self):
        return MultiplyOp(self.matrix.T, self.range, self.domain)


    """Test solutions of the linear equation Ax = b with dense A."""

def test_landweber():
    n = 3

    # Np as validation
    A = np.random.rand(n, n)
    x = np.random.rand(n)
    # Landweber is slow and needs a decent initial guess
    b = np.dot(A, x) + 0.1 * np.random.rand(n)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.element(x)
    bvec = rn.element(b)

    # Make operator
    norm = np.linalg.norm(A, ord=2)
    Aop = MultiplyOp(A)

    # Solve using landweber
    solvers.landweber(Aop, xvec, bvec, niter=n*50, omega=1/norm**2)

    assert all_almost_equal(xvec, x, places=2)

def test_conjugate_gradient():
    n = 3

    # Np as validation
    A = np.random.rand(n, n)
    x = np.random.rand(n)
    b = np.dot(A, x) + 0.1 * np.random.rand(n)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.element(x)
    bvec = rn.element(b)

    # Make operator
    Aop = MultiplyOp(A)

    # Solve using conjugate gradient
    solvers.conjugate_gradient(Aop, xvec, bvec, niter=n)

    assert all_almost_equal(xvec, x, places=2)

def test_gauss_newton():
    n = 10

    # Np as validation
    A = np.random.rand(n, n)
    x = np.random.rand(n)
    b = np.dot(A, x) + 0.1 * np.random.rand(n)

    # Vector representation
    rn = odl.Rn(n)
    xvec = rn.element(x)
    bvec = rn.element(b)

    # Make operator
    Aop = MultiplyOp(A)

    # Solve using conjugate gradient
    solvers.gauss_newton(Aop, xvec, bvec, niter=n)

    assert all_almost_equal(xvec, x, places=2)
    

if __name__ == '__main__':
    pytest.main(__file__.replace('\\','/') + ' -v')

