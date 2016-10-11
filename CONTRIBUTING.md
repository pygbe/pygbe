# Developer's Guide

Welcome to PyGBe's developer's guide! This is a place to keep track of
information that does not belong in a regular userguide.

## Contributing to PyGBe

All code must go through the pull request review procedure. If you want to add
something to PyGBe, first fork the repository. Make your changes on your fork of
the repository and then open a pull request with your changes against the main
PyGBe repository.

New features should follow the following rules:
1. PEP8 style guide should be followed
2. New features should include unit tests
3. All functions should have NumPy style doctrings. See
   [here](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt)
   for reference)

## Running the tests

### Regression tests

Once everything is installed, you can use `py.test` to run the regression tests,
located in `pygbe/tests`

To run any of the regression tests individually, you can pass the test name to
`pytest`

```python
py.test test_lysozome.py
```

or to run all of the tests just run `py.test` within the PyGBe directory

```python
py.test
```

### Convergence tests

There is a more robust set of tests located in `pygbe/tests/convergence_tests`.
These include comparisons to analytical solutions and checks to ensure
convergence over a series of finer meshes.

Note that these tests take a few hours to run. To run them, navigate to the
convergence test folder and run

```python
python run_convergence_tests.py
```

Any individual set of convergence tests can be run by specifying a given test
file, e.g.

```python
python lysozyme.py
```

or

```python
python sphere_dirichlet.py
```

## Generating documentation

PyGBe uses [`doctr`](https://github.com/gforsyth/doctr) to automatically
generate documentation using Travis CI. If you have made a number of changes to
the docs, it is best to first manually check them to make sure everything if
working as expected. Otherwise, don't worry about it. Any changes to the docs
will be automatically compiled using Sphinx and then pushed to `gh-pages` when a
PR is merged into `master`.

### Manually generating documentation

Ensure `sphinx` is installed.

```console
$ pip install sphinx
$ conda install sphinx
```

Once you have added docstrings to some new functions, first reinstall PyGBe
using either

```console
$ python setup.py install
```

or

```console
$ python setup.py develop
```

In the root of `pygbe` run

```console
$ sphinx-apidoc -f -o docs/source pygbe
```

Then enter the docs folder and run `make`

```console
$ cd docs
$ make html
```

Ensure that the docs have built correctly and that formatting, etc, is
functional by opening the local docs in your browser

```console
firefox _build/html/index.html
```

If there are any errors in the build (or warnings), then fix them. If there are
no errors and the docs look good on your local build, then you're done! Open a
PR with your changes and when it is merged, the changes to the documentation
will be automatically built and pushed by Travis to the `gh-pages` branch.
