# PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics

[![DOI_JOSS](http://joss.theoj.org/papers/10.21105/joss.00043/status.svg)](http://dx.doi.org/10.21105/joss.00043)
[![CITE_BIB](https://img.shields.io/badge/Cite%20PyGBe-bibtex-blue.svg)](http://www.doi2bib.org/#/doi/10.21105/joss.00043)

PyGBe—pronounced _pigbē_—is a Python code to apply the boundary element method
for molecular-electrostatics calculations in a continuum model. It computes
solvation energies for proteins modeled with any number of dielectric regions.
The mathematical formulation follows Yoon and Lenhoff (1990) for solving the
Poisson-Boltzmann equation of
the [implicit-solvent](https://en.wikipedia.org/wiki/Implicit_solvation) model
in integral form.

PyGBe achieves both algorithmic and hardware acceleration. The solution
algorithm uses
a [Barnes-Hut](https://en.wikipedia.org/wiki/Barnes–Hut_simulation) treecode to
accelerate each iteration of a GMRES solver to O(N logN), for N unknowns. It
exploits NVIDIA GPU hardware on the most computationally intensive parts of the
code using CUDA kernels in the treecode, interfacing with PyCUDA. Some parts of
the code are written in C++, wrapped using SWIG.

Detailed documentation is available at http://barbagroup.github.io/pygbe/docs/

## Installation

The following instructions assume that the operating system is Ubuntu. Run the
corresponding commands in your flavor of Linux to install.

### Dependencies (last tested)
* Python 3.4+
* Numpy 1.11.1
* SciPy 0.17.1+
* SWIG 3.0.8
* NVCC 7.5 
    * gcc < 4.10
* PyCUDA 2016.1.3
* matplotlib 1.5.1+ (optional, for post-processing only)

#### Python and Numpy

To install the specific version of these packages we recommend using either
[conda](http://conda.pydata.org/docs/get-started.html) or
[pip](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/).

To create a new environment for using PyGBe with `conda` you can do the
following:

```console
conda create -n pygbe python=3.5 numpy scipy swig matplotlib
source activate pygbe
```

and then proceed with the rest of the installation instructions (although note
that if you do this, `swig` is already installed.


#### SWIG

To install SWIG we recommend using either `conda`, your distribution package
manager or [SWIG's website](http://www.swig.org/download.html).

#### NVCC

[Download and install](https://developer.nvidia.com/cuda-downloads) the CUDA
Toolkit.

#### PyCUDA

PyCUDA must be installed from source. Follow the
[instructions](http://wiki.tiker.net/PyCuda/Installation) on the PyCUDA website.
We summarize the commands to install PyCUDA on Ubuntu here:

    > cd $HOME
    > mkdir src
    > cd src
    > wget https://github.com/inducer/pycuda/archive/v2016.1.2.tar.gz
    > tar -xvzf pycuda-2016.1.2.tar.gz
    > cd pycuda-2016.1.2
    > python configure.py --cuda-root=/usr/local/cuda
    > make
    > sudo make install

If you are not installing PyCUDA systemwide, do not use `sudo` to install and
simply run

    > make install

as the final command.

Test the installation by running the following:

    > cd test
    > python test_driver.py

PyGBe has been run and tested on Ubuntu 12.04, 13.10, 15.04 and 16.04 (using gcc
< 4.10).

### Installing PyGBe

Create a clone of the repository on your machine:

    > cd $HOME/src
    > git clone https://github.com/barbagroup/pygbe.git
    > cd pygbe
    > python setup.py install clean

If you are installing PyGBe systemwide (if you installed PyCUDA systemwide),
then use `sudo` on the install command

    > sudo python setup.py install clean

## Run PyGBe

PyGBe cases are divided up into individual folders. We have included a few
example problems in `examples`.

Test the PyGBe installation by running the Lysozyme (`lys`) example in the
folder `examples`. The structure of the folder is as follows:

```
lys
  ˫ lys.param
  ˫ lys.config
  ˫ built_parse.pqr
  ˫ geometry/Lys1.face
  ˫ geometry/Lys1.vert
  ˫ output/
```

To run this case, you can use

    > pygbe examples/lys

To test PyGBe-LSPR, run the single silver sphere (``lspr_silver``) example.

To run lspr cases, you can use

    > pygbe-lspr examples/lspr_silver

To run any PyGBe case, you can pass `pygbe` (or ``pygbe-lspr`` if it's a LSPR
application) a relative or an absolute path to
the problem folder.

Note that PyGBe will grab the first `param` and `config` files that it finds in
the problem folder (they don't have to share a name with the folder, but it's
helpful for organization). If you want to explicitly pass in a
different/specific `param` or `config` file, you can use the `-p` and `-c`
flags, respectively.

If you have a centralized `geometry` folder, or want to reuse existing files
without copying them, you can also pass the `-g` flag to `pygbe` to point to the
custom location. Note that this path should point to a folder which contains a
folder called `geometry`, not to the `geometry` folder itself.

For more information on PyGBe's command line interface, run

    > pygbe -h

or

    > pygbe-lspr -h

### Mesh

In the `examples` folder, we provide meshes and `.pqr` files for a few example
problems. To plug in your own protein data, download the corresponding `.pdb`
file from the Protein Data Bank, then get its `.pqr` file using any PDB to PQR
converter (there are online tools available for this). Our code interfaces with
meshes generated using
[MSMS (Michel Sanner's Molecular Surface code)](http://mgltools.scripps.edu/packages/MSMS).

The meshes for the LSPR examples and some Poisson Boltzmann that involve spheres
were generated with a script called `mesh_sphere.py` located in 
`pygbe/preprocessing_tools/`.

In [Generate meshes and pqr](http://barbagroup.github.io/pygbe/docs/mesh_pqr_setup.html) you can find detailed instructions to generate the pqr and meshes. 



### Performance:

[PyGBe Performance](https://github.com/barbagroup/pygbe/blob/master/performance/PyGBe%20Performance.ipynb)

## References

* Barnes, J. and Hut, P. (1986), "A hierarchical O(N log N) force-calculation algorithm," _Nature_, **324**: 446–449, [doi: 10.1038/324446a0](http://dx.doi.org/10.1038/324446a0)
* Yoon, B.J. and Lenhoff, A.M. (1990), "A boundary element method for molecular electrostatics with electrolyte effects," _Journal of Computational Chemistry_,
**11**(9): 1080–1086, [doi: 10.1002/jcc.540110911](http://dx.doi.org/10.1002/jcc.540110911).
* Mayergoyz, I. D. and Zhang, Z. (2007). "The computation of extinction cross sections of
resonant metallic nanoparticles subject to optical radiation", _IEEE Trans. Magn._,
**43**(4):1681–1684,[doi: 10.1109/TMAG.2007.892500](http://ieeexplore.ieee.org/document/4137779/).
* Jung, J., Pedersen, T. G., Sondergaard, T., Pedersen, K., Larsen, A. N., and Nielsen,
B. B. (2010), "Electrostatic plasmon resonances of metal nanospheres in layered
geometries", _Phys. Rev. B_, **81**(12), [doi:10.1103/PhysRevB.81.125413](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.81.125413).

### Papers published using PyGBe

* Cooper, C.D, Bardhan, J.P. and Barba, L.A. (2014), "A biomolecular electrostatics solver using Python, GPUs and boundary elements that can handle solvent-filled cavities and Stern layers," _Computer Physics Communications_, **185**(3): 720–729, [doi: 10.1016/j.cpc.2013.10.028](http://dx.doi.org/10.1016/j.cpc.2013.10.028), [arxiv:1309.4018](http://arxiv.org/abs/1309.4018)
* Cooper, C.D and Barba, L.A. (2016), "Poisson–Boltzmann model for protein–surface electrostatic interactions and grid-convergence study using the PyGBe code," _Computer Physics Communications_, **202**: 23–32, [doi: 10.1016/j.cpc.2015.12.019](http://dx.doi.org/10.1016/j.cpc.2015.12.019), [arXiv:1506.03745](http://arxiv.org/abs/1506.03745)
* Cooper, C.D, Clementi, N.C. and Barba, L.A. (2015), "Probing protein orientation near charged nanosurfaces for simulation-assisted biosensor design," _Journal of Chemical Physics_, **143**: 124709 [doi: 10.1063/1.4931113](http://dx.doi.org/10.1063/1.4931113), [arXiv:1503.08150v4](http://arxiv.org/abs/1506.03745).

### Other software

#### Poisson-Boltzmann Solvers

A few other open-source packages exist for solving implicit-solvent models of
the Poisson-Boltzmann equation.

##### Volumetric-based solvers

* [Delphi](http://compbio.clemson.edu/delphi)
* [APBS](http://www.poissonboltzmann.org/)

##### Boundary-element method

* AFMPB (both [serial](http://cpc.cs.qub.ac.uk/summaries/AEGB_v1_1.html) and [parallel](http://cpc.cs.qub.ac.uk/summaries/AEGB_v2_0.html) versions exist)
* [TABI](http://faculty.smu.edu/wgeng/research/bipb.html)


#### Nanoplasmonic Solvers

##### Boundary-element method

* [MNPBEM](http://physik.uni-graz.at/mnpbem/#1) A Matlab Toolbox

## How to cite PyGBe

If PyGBe contributes to a project that leads to a scientific publication, please cite the project.
You can use this citation or the BibTeX entry below.

> Christopher D. Cooper, Natalia C. Clementi, Gilbert Forsyth, Lorena A. Barba (2016). PyGBe: Python, GPUs and Boundary elements for biomolecular electrostatics, _J. Open Source Software_, **1**(4):43, [doi:10.21105/joss.00043](http://dx.doi.org/10.21105/joss.00043)

```console
@article{DCooper2016,
  doi = {10.21105/joss.00043},
  url = {http://dx.doi.org/10.21105/joss.00043},
  year  = {2016},
  month = {aug},
  publisher = {The Open Journal},
  volume = {1},
  number = {4},
  author = {Christopher D. Cooper and Natalia C. Clementi and Gilbert Forsyth and Lorena A. Barba},
  title = {{PyGBe}: Python,  {GPUs} and Boundary elements for biomolecular electrostatics},
  journal = {{JOSS}}
}
```
