Programmatic use of PyGBe
-----------------------------


Using PyGBe within IPython
==========================

There will be times when you want to run PyGBe within a Python interpreter,
as opposed to running on the command line. If you need to debug using ``ipdb``
or to generate figures that require several sequential runs, you're better off
scripting in Python than in Bash.

The following example would run the Lysozyme example.

.. note:: The first value in the list passed to main must be an empty string.
          The path to the example folder (``examples/lys``) is relative to
          wherever the interpreter was started. In this example, we entered the
          root of the PyGBe repo and then ran ``ipython``

.. code:: python

    from pygbe.main import main
    main(['', 'examples/lys'], log_output=False)

If it's a LSPR application, like the single silver sphere, we do:

.. code:: python

    from pygbe.lspr import main
    main(['', 'examples/lspr_silver'], log_output=False)

Useful kwargs
=============

There are a few keyword arguments you can pass to PyGBe when running from
IPython that are useful for programmatic work.

``log_output`` : default ``True``
    If you are debugging you probably don't need to the entirety of
    PyGBe output written to a log file, this will suspend that logging behavior
    and only stdout will be written to.

``return_output_fname`` : default ``False``
    If ``True`` then ``main()`` will
    return the name of the logfile created for the current run

``return_results_dict`` : default ``False``
    If ``True`` then ``main()`` will return a dictionary with the calculated
    values (if applicable) of the config file used, param file used, any
    geometry files, the path of the problem, the number of elements and any
    calculated energy quantities.

.. note:: ``return_results_dict`` will supercede both ``return_output_fname`` and ``log_output`` if used.

``field`` : default ``None``
    If you are running several runs that are nearly identical, with only a few
    changes to the configuration, rather than programmatically editing config
    files to generate each run, you can instead pass in a dictionary of the
    appropriate values.  This will circumvent the ``initialize_field`` function
    that reads from the config file.


Sample field dictionary
=======================

For reference, the config dictionary equivalent to the Lysozyme example would look like the following:

.. code:: python

    lys_field = {'E': [80, 80, 4, 80, 80, 80],
                'LorY': [2, 1, 1, 1, 1, 1],
                'Nchild': [1, 1, 3, 0, 0, 0],
                'Nparent': [0, 1, 1, 1, 1, 1],
                'charges': [0, 0, 1, 0, 0, 0],
                'child': [0, 1, 2, 3, 4],
                'coulomb': [0, 0, 1, 0, 0, 0],
                'kappa': [0.125, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12],
                'parent': ['NA', 0, 1, 2, 3, 4],
                'pot': [0, 0, 1, 0, 0, 0],
                'qfile': ['NA',
                'NA',
                '/home/gil/git/pygbe/examples/lys/lys1_charges.pqr',
                'NA'
                'NA',
                'NA']}

.. note:: For LSPR applications we have an extra keyword ``lspr_values``. This flag is 
          used to pass the incoming electric field parameters in a programmatic
          fashion. 

    ``lspr_values`` : default ``None``
    
    If you are running several runs that are nearly identical, with only a few
    changes to the electric field configuration, rather than programmatically 
    editing config files to generate each run, you can instead pass in a tuple of
    the appropriate values.  This will circumvent the ``read_electric_field`` 
    function that reads from the config file.
    
    
Sample lspr_values tuple
=========================

For reference, if we want to run the ``lspr_silver`` for different wavelengths, 
we create tuple that would look like:

.. code:: python

    lspr_values = (-1, [3800, 3850, 3900, 3950])

In this case, keep in mind that the dielectric constant in LSPR cases depends
on the wavelength. Therefore if you iterate over the wavelength you will need
to update field 'E' in your field dictionary. For example, you can create a list
where each element is a tuple of the form``(wavelength, diel_field)``. To iterate
over each element of the list you would do something like:

.. code:: python

    wave_diel = list(zip(wavelength, diel))

    for wave, E in wave_diel:
        field_dict['E'] = E  
        results = main(['', example_folder_path], field=field_dict,
                       lspr_values=(-1,wave), return_results_dict=True)
