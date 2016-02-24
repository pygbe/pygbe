import sys
from setuptools import setup, find_packages, Extension
import numpy

def main():
    if sys.version_info[0] != 2:
        sys.exit('PyGBe only supports Python 2.7')

    setupkw = dict(
            name='PyGBe',
            description='A boundary element method code that does molecular electrostatics calculations with a continuum approach',
            platforms='Linux',
            packages = find_packages(),
            ext_modules = [
                Extension("_multipole",
                          sources=["pygbe/tree/multipole.i", "pygbe/tree/multipole.cpp"],
                          swig_opts=['-c++'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=['-fPIC', '-O3', '-funroll-loops', '-msse3', '-fopenmp'],
                ),
                Extension("_direct",
                          sources=["pygbe/tree/direct.i", "pygbe/tree/direct.cpp"],
                          swig_opts=['-c++'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=['-fPIC', '-O3', '-funroll-loops', '-msse3', '-fopenmp'],
                ),
                Extension("_calculateMultipoles",
                          sources=["pygbe/tree/calculateMultipoles.i", "pygbe/tree/calculateMultipoles.cpp"],
                          swig_opts=['-c++'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=['-fPIC', '-O3', '-funroll-loops', '-msse3', '-fopenmp'],
                ),
                Extension("_semi_analyticalwrap",
                          sources=["pygbe/util/semi_analyticalwrap.i", "pygbe/util/semi_analyticalwrap.cpp"],
                          swig_opts=['-c++'],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=['-fPIC', '-O3', '-funroll-loops', '-msse3', '-fopenmp'],
                ),
                ]
            )
    setup(**setupkw)

if __name__ == '__main__':
    main()
