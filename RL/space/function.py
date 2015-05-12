""" Module for spaces whose elements are Functionals
"""
# Copyright 2014, 2015 Holger Kohr, Jonas Adler
#
# This file is part of RL.
#
# RL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RL.  If not, see <http://www.gnu.org/licenses/>.


# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from future import standard_library
try:
    from builtins import super
except ImportError:  # Versions < 0.14 of python-future
    from future.builtins import super

# RL imports
import RL.operator.functional as fun
from RL.space.space import HilbertSpace, Algebra

from RL.space.set import RealNumbers, Set
from RL.utility.utility import errfmt

standard_library.install_aliases()


# Example of a space:
class FunctionSpace(Algebra):
    """ The space of scalar valued functions on some domain

    Parameters
    ----------

    domain : Set
             The set the functions take values from
    field : {RealNumbers, ComplexNumbers}, optional
            The field that the functions map values into.
            Since FunctionSpace is a LinearSpace, this is also
            the set of scalars for this space.
    """

    def __init__(self, domain, field=None):
        if not isinstance(domain, Set):
            raise TypeError("domain ({}) is not a set".format(domain))

        self.domain = domain
        self._field = field if field is not None else RealNumbers()

    def linCombImpl(self, a, x, b, y):
        """ Returns a function that calculates (a*x + b*y)(t) = a*x(t) + b*y(t)

        The created object is rather slow, and should only be used for testing purposes.
        """
        return a*x + b*y  # Use operator overloading

    def multiplyImpl(self, x, y):
        """ Returns a function that calculates (x * y)(t) = x(t) * y(t)
        
        The created object is rather slow, and should only be used for testing purposes.
        """
        return self.makeVector(lambda *args: x(*args)*y(*args))

    @property
    def field(self):
        """ The field that the functions map values into.

        Since FunctionSpace is a LinearSpace, this is also
        the set of scalars for this space.
        """
        return self._field

    def equals(self, other):
        """ Verify that other is a FunctionSpace with the same domain and field
        """
        return isinstance(other, FunctionSpace) and self.domain == other.domain and self.field == other.field

    def empty(self):
        """ Returns the zero function (the function which maps any value to zero)
        """
        return self.makeVector(lambda *args: 0)

    def zero(self):
        """ Returns the zero function (the function which maps any value to zero)
        """
        return self.makeVector(lambda *args: 0)

    def makeVector(self, function):
        """ Creates an element in FunctionSpace

        Parameters
        ----------
        function : Function from self.domain to self.field
                   The function that should be converted/reinterpreted as a vector.

        Returns
        -------
        FunctionSpace.Vector instance


        Examples
        --------

        >>> R = RealNumbers()
        >>> space = FunctionSpace(R, R) 
        >>> x = space.makeVector(lambda t: t**2)
        >>> x(1)
        1.0
        >>> x(3)
        9.0
        """
        return FunctionSpace.Vector(self, function)

    def __str__(self):
        return "FunctionSpace " + str(self.domain) + "->" + str(self.field)

    def __repr__(self):
        return "FunctionSpace(" + str(self.domain) + ", " + str(self.field) + ")"

    class Vector(Algebra.Vector, fun.Functional):
        """ A Vector in a FunctionSpace

        FunctionSpace-Vectors are themselves also Functionals, and inherit
        a large set of features from them.

        Parameters
        ----------

        space : FunctionSpace
                Instance of FunctionSpace this vector lives in
        values : Function from space.domain to space.field
                 The function that should be converted/reinterpreted as a vector.
        """

        def __init__(self, space, function):
            super().__init__(space)
            self.function = function

        def applyImpl(self, rhs):
            """ Apply the functional in some point
            """
            return self.function(rhs)

        @property
        def domain(self):
            """ The range of this Vector (when viewed as a functional)
            """
            return self.space.domain

        @property
        def range(self):
            """ The range of this Vector (when viewed as a functional)

            The range is the same as the field of the vectors space
            """
            return self.space.field

        def __str__(self):
            return str(self.function)

        def __repr__(self):
            return repr(self.space) + '.makeVector(' + repr(self.function) + ')'


class L2(FunctionSpace, HilbertSpace):
    """The space of square integrable functions on some domain
    """

    def __init__(self, domain, field=None):
        super().__init__(domain, field)

    def innerImpl(self, v1, v2):
        """ TODO: remove?
        """
        raise NotImplementedError(errfmt('''
        You cannot calculate inner products in non-discretized spaces'''))

    def equals(self, other):
        """ Verify that other is equal to this space as a FunctionSpace and also a L2 space.
        """
        return isinstance(other, L2) and FunctionSpace.equals(self, other)

    def __str__(self):
        return "L2 " + str(self.domain) + "->" + str(self.field)

    def __repr__(self):
        return "L2(" + str(self.domain) + ", " + str(self.field) + ")"

    class Vector(FunctionSpace.Vector, HilbertSpace.Vector):
        pass

if __name__ == '__main__':
    import doctest
    doctest.testmod()