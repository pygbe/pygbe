PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics
=========================================================================

PyGBe—pronounced *pigbē*—is a Python code to apply the boundary element
method for molecular-electrostatics calculations in a continuum model.
It computes solvation energies for proteins modeled with any number of
dielectric regions. The mathematical formulation follows Yoon and
Lenhoff (1990) for solving the Poisson-Boltzmann equation of the
`implicit-solvent <https://en.wikipedia.org/wiki/Implicit_solvation>`__
model in integral form.

PyGBe achieves both algorithmic and hardware acceleration. The solution
algorithm uses a
`Barnes-Hut <https://en.wikipedia.org/wiki/Barnes–Hut_simulation>`__
treecode to accelerate each iteration of a GMRES solver to O(N logN),
for N unknowns. It exploits NVIDIA GPU hardware on the most
computationally intensive parts of the code using CUDA kernels in the
treecode, interfacing with PyCUDA. Some parts of the code are written in
C++, wrapped using SWIG.
