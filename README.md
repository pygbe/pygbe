# PyGBe: Python GPU code for Boundary elements

This is a boundary element method code that does molecular electrostatics 
calculations with a continuum approach. It calculates solvation energies for 
proteins modeled with any number of dielectric regions. We use the formulation 
presented in the paper by Yoon and Lenhoff: ["A Boundary Element Method for Molecular 
Electrostatics with Electrolyte Effects", Journal of Computational Chemistry, 
Vol. 11, No. 9, 1990](http://dx.doi.org/10.1002/jcc.540110911). Proper user guide is under development.

This code is accelerated using the Barnes-Hut treecode so that each GMRES 
iteration scales as O(NlogN).
The code is written in Python putting the most computationally intensive 
parts on the GPU, interfacing with PyCUDA, and some parts are wrapped in 
C++ using SWIG. 

## Installation

The following instructions assume that the operating system is Ubuntu. Run the 
corresponding commands in your flavor of Linux to install.

### Dependencies (last tested)
* Python 2.7.11
* Numpy 1.10.4
* SWIG 3.0.8
* NVCC 7.0 
* PyCUDA 2015.1.3

#### Python and Numpy 

To install the specific version of these packages we recommend using either [conda](http://conda.pydata.org/docs/get-started.html) or [pip](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/).

#### SWIG

To install SWIG we recommend using either `conda`, your distribution package manager or [SWIG's website](http://www.swig.org/download.html).  

#### NVCC

[Download and install](https://developer.nvidia.com/cuda-downloads) the CUDA Toolkit.

#### PyCUDA

PyCUDA must be installed from source. Follow the [instructions](http://wiki.tiker.net/PyCuda/Installation) on the PyCUDA website.
We summarize the commands to install PyCUDA on Ubuntu here:

    > cd $HOME
    > mkdir src
    > cd src
    > wget https://pypi.python.org/packages/source/p/pycuda/pycuda-2015.1.3.tar.gz
    > tar -xvzf pycuda-2015.1.3.tar.gz
    > cd pycuda-2015.1.3
    > python configure.py --cuda-root=/usr/local/cuda
    > make
    > sudo make install 
    
If you are not installing PyCUDA systemwide, do not use `sudo` to install and simply run

    > make install
    
as the final command.

Test the installation by running the following:

    > cd test
    > python test_driver.py

PyGBe has been run and tested on Ubuntu 12.04, 13.10 and 15.04. 

### Installing PyGBe

Create a clone of the repository on your machine:

    > cd $HOME/src
    > git clone https://github.com/barbagroup/pygbe.git
    > cd pygbe
    > python setup.py install clean
    
If you are installing PyGBe systemwide (if you installed PyCUDA systemwide), then use `sudo` on the install command

    > sudo python setup.py install clean

### Run PyGBe

PyGBe cases are divided up into individual folders.  We have included a few example problems in `examples`.  Additional problem folders can be downloaded from [coming soon]().

Test the PyGBe installation by running the Lysozyme (`lys`) example in the folder `examples`.
The structure of the folder is as follows:

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
    
To run any PyGBe case, you can pass `pygbe` a relative or an absolute path to the problem folder. 

If you have a centralized `geometry` folder, or want to reuse existing files without copying them, you can also pass the `-g` flag to `pygbe` to point to the custom location.  Note that this path should point to a folder which contains a folder called `geometry`, not to the `geometry` folder itself.

### Mesh
In the `examples` folder, we provide meshes and `.pqr` files for a few example problems. 
To plug in your own protein data, download the corresponding `.pdb` file from the Protein Data Bank, 
then get its `.pqr` file using any PDB to PQR converter (there are online tools available for this). 
Our code interfaces with meshes generated using [MSMS (Michel Sanner's 
Molecular Surface code)](http://mgltools.scripps.edu/packages/MSMS).  

Let us know if you have any questions/feedback.

Enjoy!

Christopher (cdcooper@bu.edu)
