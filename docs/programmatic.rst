Programmatic use of ``PyGBe``
-----------------------------


Using ``PyGBe`` within IPython
==============================


.. code:: python

   from pygbe.main import main
    main(['', 'examples/lys'], log_output=False)

   




.. code:: console

    conda create -n pygbe python=3.5 numpy scipy swig matplotlib
    source activate pygbe

