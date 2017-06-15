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

PyGBe—pronounced _pigbē_—is a Python library that uses the boundary integral 
method applied to biomolecular electrostatics and nanoparticle plasmonics. 

This PyGBe release updates the exisiting library presented in @DCooper2016 to Python 3,
introduces an application to nanoplasmonics and, it includes better regression tests
using pytest and a redesign of the convergence tests.

The nanoplasmonics incorporation allows treating localized surface plasmons resonance
quasi-statically (see @Mayergoyz2007). Localized surface plasmon resonance (LSPR) is an optical
effect (see @Bohren1983), but electrostatics is a good approximation in the long-wavelength
limit. We use an integral formulation (see @Jung2010), making the existing Boundary element 
approach suitable and able to exploits the exisiting algorithmic and hardware 
accelaration detailed in @CooperBardhanBarba2014.

For nanoparticles smaller than the wavelength of incident light, PyGBe 
can compute the extinction cross-section in absorbing and non-absorbing mediums
@Mishchenko2007. We plan to use this new feature of PyGBe to study the 
suitability and performance of nanobiosensors and, to explore nanophotonics 
applications.

We believe PyGBe to be the first open-source software able to compute extinction
cross-sections of arbitrary geometry. 

# References
