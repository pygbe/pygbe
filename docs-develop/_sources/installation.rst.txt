Installation
------------

Regular installation
~~~~~~~~~~~~~~~~~~~~

The following instructions assume that the operating system is Ubuntu.
Run the corresponding commands in your flavor of Linux to install.

Dependencies (last tested)
^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Python 3.4+ (3.6.1)
-  Numpy 1.11.1+ (1.13.1)
-  SciPy 0.17.1+ (0.19.1)
-  SWIG 3.0.8+ (3.0.10)
-  NVCC 8.0

   -  gcc  5.4.0

-  PyCUDA 2017.1.1
-  matplotlib 1.5.1+ (2.0.2) (optional, for post-processing only)

Python and Numpy
^^^^^^^^^^^^^^^^

To install the specific version of these packages we recommend using
either `conda <http://conda.pydata.org/docs/get-started.html>`__ or
`pip <http://python-packaging-user-guide.readthedocs.org/en/latest/installing/>`__.

To create a new environment for using PyGBe with ``conda`` you can do
the following:

.. code:: console

    conda create -n pygbe python=3.6 numpy scipy swig matplotlib
    source activate pygbe

and then proceed with the rest of the installation instructions
(although note that if you do this, ``swig`` is already installed.

SWIG
^^^^

To install SWIG we recommend using either ``conda``, your distribution
package manager or `SWIG's
website <http://www.swig.org/download.html>`__.

NVCC
^^^^

`Download and install <https://developer.nvidia.com/cuda-downloads>`__
the CUDA Toolkit.

PyCUDA
^^^^^^

PyCUDA must be installed from source. Follow the
`instructions <http://wiki.tiker.net/PyCuda/Installation>`__ on the
PyCUDA website. We summarize the commands to install PyCUDA on Ubuntu
here:

::

    > cd $HOME
    > mkdir src
    > cd src
    > wget https://github.com/inducer/pycuda/archive/v2016.1.2.tar.gz
    > tar -xvzf pycuda-2016.1.2.tar.gz
    > cd pycuda-2016.1.2
    > python configure.py --cuda-root=/usr/local/cuda
    > make
    > sudo make install

If you are not installing PyCUDA systemwide, do not use ``sudo`` to
install and simply run

::

    > make install

as the final command.

Test the installation by running the following:

::

    > cd test
    > python test_driver.py


Installing PyGBe
~~~~~~~~~~~~~~~~

Create a clone of the repository on your machine:

::

    > cd $HOME/src
    > git clone https://github.com/barbagroup/pygbe.git
    > cd pygbe
    > python setup.py install clean

If you are installing PyGBe systemwide (if you installed PyCUDA
systemwide), then use ``sudo`` on the install command

::

    > sudo python setup.py install clean

PyGBe has been run and tested on Ubuntu 12.04, 13.10, 15.04 and 16.04.


Installation using `Docker <https://docs.docker.com/get-started/>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Requirements
^^^^^^^^^^^^
- Install `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__, (instructions in their README)

    - Check `pre-requisites <https://github.com/NVIDIA/nvidia-docker/wiki/Installation#prerequisites>`__

- Follow instructions at the top of ``Dockerfile``.
