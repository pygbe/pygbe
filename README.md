# PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics

[![DOI_JOSS](http://joss.theoj.org/papers/10.21105/joss.00043/status.svg)](http://dx.doi.org/10.21105/joss.00043)
[![CITE_BIB](https://img.shields.io/badge/Cite%20PyGBe-bibtex-blue.svg)](http://www.doi2bib.org/#/doi/10.21105/joss.00043)

PyGBe—pronounced _pigbē_—is a Python code to apply the boundary element method for molecular-electrostatics
calculations in a continuum model.
It computes solvation energies for proteins modeled with any number of dielectric regions.
The mathematical formulation follows Yoon and Lenhoff (1990) for solving the Poisson-Boltzmann equation of the [implicit-solvent](https://en.wikipedia.org/wiki/Implicit_solvation) model in integral form.

PyGBe achieves both algorithmic and hardware acceleration.
The solution algorithm uses a [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes–Hut_simulation) treecode to accelerate each iteration of a GMRES solver to O(N logN), for N unknowns.
It exploits NVIDIA GPU hardware on the most computationally intensive parts of the code using CUDA kernels in the treecode, interfacing with PyCUDA.
Some parts of the code are written in C++, wrapped using SWIG.

Please visit http://barbagroup.github.io/pygbe/docs/index.html for more information.
