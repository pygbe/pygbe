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
