PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics and nanoparticle plasmonics 
=====================================================================================================

PyGBe—pronounced *pigbē*—is a Python library that applies the boundary integral 
method for biomolecular electrostatics and nanoparticle plasmonics. 

Biomolecular electrostatics:
----------------------------

For this application PyGBe computes solvation energies for proteins modeled with
any number of dielectric regions. It uses uses a continuum model and the 
mathematical formulation follows Yoon and Lenhoff (1990) for solving the 
Poisson-Boltzmann equation of the `implicit-solvent <https://en.wikipedia.org/wiki/Implicit_solvation>`__
model in integral form.

Localized Surface Plasmon Resonance :
-------------------------------------

For this application PyGBe computes the extinction cross section for nanoparticles, 
handeling the localized surface plasmon effects quasi-statically. It requires the
the nanoparticles to be smaller than the wavelength of the incident light 
(long-wavelength limit) where electrostatics is a good approximation 
(`Mayergoyz, I. D. and Zhang, Z. (2007) <http://ieeexplore.ieee.org/abstract/document/4137779>`__,
`Jung, J., et al (2010) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.125413>`__).


PyGBe achieves both algorithmic and hardware acceleration. The solution
algorithm uses a
`Barnes-Hut <https://en.wikipedia.org/wiki/Barnes–Hut_simulation>`__
treecode to accelerate each iteration of a GMRES solver to O(N logN),
for N unknowns. It exploits NVIDIA GPU hardware on the most
computationally intensive parts of the code using CUDA kernels in the
treecode, interfacing with PyCUDA. Some parts of the code are written in
C++, wrapped using SWIG.
