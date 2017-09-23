# Examples

This directory contains example code demonstrating ODL functionality.

## Content

* [deform](deform) Creating and using deformation operators in the `odl.deform` package.
* [diagnostics](diagnostics) Testing operators and spaces with `OperatorTest` and `SpaceTest`.
* [operator](operator) Examples for subclassing `Operator`.
* [solvers](solvers) Usage examples for solvers in the odl.solvers package. Some of the examples use a wide variety of ODL features, from creating a forward operator, e.g., a `RayTransform`, to solving an inverse problem.
* [space](space) Demonstrations of general functionality of spaces, and creation of a new space.
* [tomo](tomo) How to create and call a `RayTransform` for various geometries, how to use analytic reconstruction methods like FBP, and a performance comparison.
* [trafos](trafos) Showcase of `FourierTransform` and `WaveletTransform`. See `solvers` for use cases in inverse problems.
* [ufunc_ops](ufunc_ops) Examples on how to use functions like `numpy.sin` as an operator.
* [visualization](visualization) Demonstration of the visualization functionality in ODL, including 1d, 2d and slice views, as well as real-time updates.
