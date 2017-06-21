PyGBe: Python and GPU Boundary-integral solver for electrostatics
=================================================================

PyGBe—pronounced *pigbē*—is a Python library that applies the boundary integral 
method for biomolecular electrostatics and nanoparticle plasmonics.

PyGBe achieves both algorithmic and hardware acceleration. The solution
algorithm uses a
`Barnes-Hut <https://en.wikipedia.org/wiki/Barnes–Hut_simulation>`__
treecode to accelerate each iteration of a GMRES solver to O(N logN),
for N unknowns. It exploits NVIDIA GPU hardware on the most
computationally intensive parts of the code using CUDA kernels in the
treecode, interfacing with PyCUDA. Some parts of the code are written in
C++, wrapped using SWIG.

Biomolecular electrostatics:
----------------------------

In this application, PyGBe uses continuum electrostatics to compute the solvation
energy for proteins modeled with any number of dielectric regions. The 
mathematical formulation follows Yoon and Lenhoff (1990) for solving the 
Poisson-Boltzmann equation of the `implicit-solvent <https://en.wikipedia.org/wiki/Implicit_solvation>`__
model in integral form.

Localized Surface Plasmon Resonance :
-------------------------------------

PyGBe also uses electrostatics to compute the extinction cross section of 
scatterers that are much smaller than the incident wavelength. This is relevant, 
for example, to model localized surface plasmon resonance of nanoparticles, where
the quasi-static approximation is valid
(`Mayergoyz, I. D. and Zhang, Z. (2007) <http://ieeexplore.ieee.org/abstract/document/4137779>`__,
`Jung, J., et al (2010) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.125413>`__).



