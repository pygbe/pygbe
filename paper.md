---
title: 'PyGBe-LSPR: Python and GPU Boundary-integral solver for electrostatics'
tags:
  - electrostatics
  - biophysics
  - Poisson-Boltzmann
  - nano-plasmonics
  - lspr
authors:
 - name: Natalia C. Clementi
   orcid: 0000-0002-0575-5520
   affiliation: 1
 - name: Gilbert Forsyth
   orcid: 0000-0002-4983-1978
   affiliation: 1
 - name: Christopher D. Cooper
   orcid: 0000-0003-0282-8998
   affiliation: 2
 - name: Lorena A. Barba
   orcid: 0000-0001-5812-2711
   affiliation: 1
affiliations:
 - name: The George Washington University
   index: 1
 - name: Universidad Técnica Federico Santa María
   index: 2
date: 12 June 2017
bibliography: paper.bib
---

# Summary

PyGBe—pronounced _pigbē_—is a Python library for applications in
biomolecular electrostatics and nanoparticle plasmonics.
The previous code release, reported in @DCooper2016, solves the Poisson-Boltzmann equation
for biomolecules immersed in an ionic solvent, using the boundary integral method.
It computes the solvation energy, which is the free energy spent in moving a biomolecule
from vacuum to its dissolved state.
This quantity is used for assessing binding affinity, protein-surface interactions
(@CooperClementiBarba2015), and other mechanisms at this scale.

This PyGBe release makes the following contributions:
(1) it updates the exisiting library presented in @DCooper2016 to Python 3,
(2) it introduces a new capability to solve problems in nanoplasmonics, and
(3) it includes better regression tests using pytest and a redesign of the convergence tests.


The largest contribution in this release is extending PyGBe to nanoplasmonics,
by treating localized surface plasmon resonance (LSPR) quasi-statically (see @Mayergoyz2007).
LSPR is essentially a miniaturization of SPR: the resonance of the electron cloud on a
metallic surface, excited by incident light.
It is an optical effect (see @Bohren1983), but electrostatics is a good approximation in the
long-wavelength limit. This leads to a coupled system of Poisson equations on complex dielectric regions.
We use an integral formulation (see @Jung2010), making the existing boundary integral
approach suitable.
The code exploits algorithmic speedup via the Barnes-Hut treecode (@BarnesHut1986),
as detailed in @CooperBardhanBarba2014.
The complex scenario required adapting the linear solver (a GMRES algorithm),
modifying the right-hand side, and being able to use the existing treecode
separately on the real and imaginary parts of the resulting system.

PyGBe's LSPR computations measure the scattered electromagnetic field on a detector
that is located far away from a nanoparticle.
For nanoparticles smaller than the wavelength of incident light, PyGBe
can compute the extinction cross-section of absorbing and non-absorbing media
@Mishchenko2007.

To our knowledge, PyGBe is the only open-source software that uses a fast algorithm—O(N logN),
for N unknowns—and hardware acceleration on GPUs to compute the extinction cross-sections
of arbitrary geometries. We plan to use PyGBe-LSPR research related to nanobiosensors and to explore
nanophotonics applications.


# References
