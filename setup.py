import sys
from setuptools import setup, find_packages, Extension
def main():
    if sys.version_info[0] != 2:
        sys.exit('PyGBe only supports Python 2.7')

    setupkw = dict(
            name='PyGBe',
            description='A boundary element method code that does molecular electrostatics calculations with a continuum approach',
            platforms='Linux',
            packages = find_packages(),
            )
    setup(**setupkw)

if __name__ == '__main__':
    main()
