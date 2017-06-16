---
title: 'PyGBe: Python and GPU Boundary-integral solver for electrostatics'
tags:
  - electrostatics
  - biophysics
  - Poisson-Boltzmann
  - nano-plasmonics
  - lspr
authors:
 - name: Natalia C. Clementi
   orcid: 0000-0002-0575-5520
   affiliation: The George Washington University
 - name: Gilbert Forsyth
   orcid: 0000-0002-4983-1978
   affiliation: The George Washington University
 - name: Christopher D. Cooper
   orcid: 0000-0003-0282-8998
   affiliation: Universidad Técnica Federico Santa María
 - name: Lorena A. Barba
   orcid: 0000-0001-5812-2711
   affiliation: The George Washington University
date: 12 June 2017
bibliography: paper.bib
---

# Summary

PyGBe—pronounced _pigbē_—is a Python library that solves problems in 
biomolecular electrostatics and nanoparticle plasmonics
using the boundary integral method.

This PyGBe release makes the followign contributions:
(1) it updates the exisiting library presented in @DCooper2016 to Python 3,
(2) it introduces a new capability to solve problems in nanoplasmonics, and 
(3) it includes better regression tests using pytest and a redesign of the convergence tests.


The largest contribution is extending PyGBe to nanoplasmonics, 
by treating localized surface plasmon resonance (LSPR)
quasi-statically (see @Mayergoyz2007). LSPR is an optical
effect (see @Bohren1983), but electrostatics is a good approximation in the long-wavelength
limit. We use an integral formulation (see @Jung2010), making the existing boundary
approach suitable and able to exploit the algorithmic and hardware 
accelaration detailed in @CooperBardhanBarba2014, via the [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes–Hut_simulation) treecode (@BarnesHut1986).
For nanoparticles smaller than the wavelength of incident light, PyGBe 
can compute the extinction cross-section of absorbing and non-absorbing media
@Mishchenko2007. 

To our knowledge, PyGBe is the first fully open software for computing extinction cross-sections of arbitrary geometries. 
It is also the only such software (again, as far as we know) that uses a fast algorithm—O(N logN), for N unknowns—and hardware acceleration on GPUs. 
A Matlab toolbox is available, [MNPBEM](http://physik.uni-graz.at/mnpbem/#1), that provides similar functionality, but with poor scaling (N^2 or even N^3).
It also stores the full matrix, whereas PyGBe is matrix free.
Nevertheless, MNPBEM has found seveal applicantions in nanoplasmonics.
We plan to use PyGBe-LSPR to the study of nanobiosensors and to explore nanophotonics applications.

# References
