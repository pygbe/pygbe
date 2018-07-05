Input File Format
-----------------

In this file, the format for the input files is detailed. To run, PyGBe needs
two input files: a ``config`` file and a ``parameters`` file. You can find
samples of these files in the ``examples`` directory.

Config file format
~~~~~~~~~~~~~~~~~~

This file has two parts:

``FILE:`` indicates in that line a mesh file will be specified. This
code interfaces with the MSMS format, which generates a
``filename.vert`` and ``filename.face`` files. Each ``filename`` has the
information for the mesh of one surface. To account for more than one
surface (ie. for Stern layers, solvent filled cavities, several
proteins), more than one ``FILE`` line is needed. After ``filename``,
the user must specify what kind of surface ``filename`` is. It can be:

-  ``stern_layer``: surface interfacer stern layer with solvent.

-  ``dielectric_interface``: 

        -  In Poisson Boltzmann applications: surface interfaces low dielectric
           (inside protein) and high dielectric (outside protein).
        -  In LSPR applications: surface interfaces dielectric inside the 
           particle and the dielectric of the medium.

-  ``internal_cavity``: surface is an internal cavity (a result of
   '-all\_components' flag in MSMS). This is important to specify
   because by default MSMS changes the vertex ordering for internal
   cavities.
-  ``dirichlet_surface``: surface of specified potential. The value of
   this potential is read from a text file which has to be specified
   next to 'dirichlet\_surface'. (See .phi0 file in example).

-  ``neumann_surface``: surface of specified potential. The value of
   this potential is read from a text file which has to be specified
   next to 'neumann\_surface'. (See .phi0 file in example).

``FIELD:`` specifies physical parameters of the different regions.
Parameters:

-  ``LorY``: 1: Laplace, 2: Yukawa

-  ``E?``: 1: calculate the energy in this region 0: don't calculate
   energy in this region Note: if region is surrounded by a dirichlet or
   neumann surface, surface energy will be calculated.

-  ``Dielec``: Dielectric constant

-  ``kappa``: reciprocal of Debye length

-  ``charges?``: 0: No charges inside this region 1: There are charges
   in this region

-  ``coulomb?``: 0: don't calculate coulomb energy in this region 1:
   calculate coulomb energy in this region

-  ``charge_file``: location of the '.pqr' file with the location of the
   charges

-  ``Nparent``: Number of 'parent' surfaces (surface containing this
   region) Of course, this is either 1 of 0 (if it corresponds to the
   infinite region)

-  ``parent``: file of the parent surface mesh, according to their
   position in the FILE list, starting from 0 (eg. if the mesh file for
   the parent is the third one specified in the FILE section, parent=2)

-  ``Nchild``: number of child surfaces (surfaces completely contained
   in this region).

-  ``children``: position of the mesh files for the children surface in
   the FILE section

``WAVE:`` Only applicable in LSPR problems. It specifies physical parameters of
the incoming electric field in LSPR applications.
Parameters:

-  ``Efield``: electric field intensity, it is in the 'z' direction, '-' 
   indicates '-z'.

-  ``Wavelength``: wavelength of the incident electric field, in Ångström.

Parameters file format
----------------------

-  ``Precision``: double or float. (float not supported yet!).

-  ``K``: number of Gauss points per element (1, 3, 4, and 7 are
   supported).

-  ``Nk``: number of Gauss points per triangle edge for semi-analytical
   integration.

-  ``K_fine``: number of Gauss points per element for near singular
   integrals.

-  ``threshold``: defines region near singularity where semi-analytical
   technique is used. if sqrt(2\*Area)/r > threshold, integration is
   done semi-analytically.

-  ``BSZ``: CUDA block size.

-  ``restart``: number of iterations for GMRES to do restart.

-  ``tolerance``: GMRES tolerance.

-  ``max_iter``: maximum number of GMRES iterations.

-  ``P``: order of expansion in treecode.

-  ``eps``: epsilon machine.

-  ``NCRIT``: maximum number of boundary elements per twig box of tree
   structure.

-  ``theta``: multipole acceptance criterion of treecode.

-  ``GPU``: 0: don't use GPU. 1: use GPU.
