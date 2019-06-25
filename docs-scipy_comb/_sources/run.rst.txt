Run PyGBe
---------

PyGBe cases are divided up into individual folders. We have included a
few example problems in ``examples``.

Test the PyGBe installation by running the Lysozyme (``lys``) example in
the folder ``examples``. The structure of the folder is as follows:

::

    lys
      ˫ lys.param
      ˫ lys.config
      ˫ built_parse.pqr
      ˫ geometry/Lys1.face
      ˫ geometry/Lys1.vert
      ˫ output/

To run this case, you can use

::

    > pygbe examples/lys

To test PyGBe-LSPR, run the single silver sphere (``lspr_silver``) example.

To run lspr cases, you can use

::

    > pygbe-lspr examples/lspr_silver

To run any PyGBe case, you can pass ``pygbe`` (or ``pygbe-lspr`` if it's a LSPR
application) a relative or an absolute path to the problem folder.

Note that PyGBe will grab the first ``param`` and ``config`` files that
it finds in the problem folder (they don't have to share a name with the
folder, but it's helpful for organization). If you want to explicitly
pass in a different/specific ``param`` or ``config`` file, you can use
the ``-p`` and ``-c`` flags, respectively.

If you have a centralized ``geometry`` folder, or want to reuse existing
files without copying them, you can also pass the ``-g`` flag to
``pygbe`` to point to the custom location. Note that this path should
point to a folder which contains a folder called ``geometry``, not to
the ``geometry`` folder itself.

For more information on PyGBe's command line interface, run

::

    > pygbe -h

or

::

    > pygbe-lspr -h

Mesh:
~~~~~

In the ``examples`` folder, we provide meshes and ``.pqr`` files for a
few example problems. To plug in your own protein data, download the
corresponding ``.pdb`` file from the Protein Data Bank, then get its
``.pqr`` file using any PDB to PQR converter (there are online tools
available for this). Our code interfaces with meshes generated using
`MSMS (Michel Sanner's Molecular Surface
code) <http://mgltools.scripps.edu/packages/MSMS>`__.

The meshes for the LSPR examples and some Poisson Boltzmann that involve spheres,
where generated with a script called ``mesh_sphere.py`` located in 
``pygbe/preprocessing_tools/``.

In `Generate meshes and pqr <http://barbagroup.github.io/pygbe/docs/mesh_pqr_setup.html>`__ you can find detailed instructions to generate the pqr and meshes.  


Input and Parameter files:
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can find detailed instructions to generate these files in the `Input Files section <http://barbagroup.github.io/pygbe/docs/input_format.html>`__.


Performance:
~~~~~~~~~~~~

A short notebook outlining performance gains vs APBS is available here:
`PyGBe
Performance <https://github.com/barbagroup/pygbe/blob/master/performance/PyGBe_Performance.ipynb>`__
