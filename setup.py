import sys
from distutils.command.build import build
from setuptools.command.install import install
from setuptools import setup, find_packages, Extension
import numpy

class CustomBuild(build):
    """
    Subclasses build command to ensure that built_ext is run before files
    are copied.  Make setuptools compile the SWIG files first, then run install
    """
    def run(self):
        self.run_command('build_ext')
        build.run(self)

class CustomInstall(install):
    """
    Subclasses install command to ensure that built_ext is run before files
    are copied.  Make setuptools compile the SWIG files first, then run install
    """
    def run(self):
        self.run_command('build_ext')
        self.do_egg_install()

def main():
    if sys.version_info[0] != 2:
        sys.exit('PyGBe only supports Python 2.7')

    setupkw = dict(
            name='PyGBe',
            description='A boundary element method code that does molecular electrostatics calculations with a continuum approach',
            platforms='Linux',
            install_requires = [
                'pycuda >= 2013.1',
                'numpy > 1.8',
            ],
            license='MIT',
            packages = find_packages(),
            #tell setuptools to use the custom build and install classes
            cmdclass={'build': CustomBuild, 'install': CustomInstall},
            #create an entrance point that points to pygbe.main.main
            entry_points={'console_scripts': ['pygbe = pygbe.main:main']},
            #SWIG modules with all compilation options
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
                ],
            )
    setup(**setupkw)

if __name__ == '__main__':
    main()
