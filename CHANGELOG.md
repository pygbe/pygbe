# PyGBe Change Log
----
## Current development

###Added

* Script to generate ellipsoidal meshes using icosphere 

### Fixed

* CPU and GPU RHS function on matrixfree bug (missing cross terms when multiple surfaces)
* LSPR 7 spheres regression test (needed fix due to RHS bug)    
---
### Added

### Changed

### Fixed
- Deprecated import  
      `scipy.misc.comb deprecated` -> `scipy.special.comb`
### Removed

## 0.3.2
---
### Added

* Notes to Dockerfile on how to run.
* `pytest` specific version to Dockerfile otherwise pycuda won't installed properly 

### Changed

* Log output to include information of the wavelength and electric field for lspr cases.
* How file-name is written  to include wavelength information for lspr cases

### Fixed

* CPU bug in matrixfree RHS function
* Docsrings problem with sphinxs
* Typo "thresold" in param files. 

### Removed


## 0.3.1
---
### Added

* Suggestions from JOSS reviewers.
* Dockerfile for alternative installation.

### Changed

* README with latest version of packages tested.
  * Support for `CUDA 8.0` using `gcc 5.4.0` and, PyCUDA 2017.1.1.

### Fixed

* Problem with format of files needed to run performance check.
* Broken link in documentation.
* Small bug: overwriting a variable doesn't allow casting type. 

### Removed


## 0.3
---
### Added

* Ported PyGBe to Python 3 (!!!).  This breaks Python 2 support, but who cares.
* Better regression tests (faster, anyway) using pytest
* Localized Surface Plasmon Resonance application.
* lspr main function to run this applications separately. 
* lspr application's examples to the `examples` folder.
* Regression and convergence tests for the examples added. 
* Update documentation with new application to github pages

### Changed

* Old regression test suite renamed to `convergence_tests`
* All surface related functions are now methods of the surface class
* All field related functions are now methods of the field class
* Use scipy.constants instead of hardcoded values
* Docstrings use proper references now (looks better on Sphinx)
* GMRES function to accept complex numbers. Changed based on `gmres_mgs`
from PyAMG, where modified Gram-Schmidt is used to orthogonalize the
Krylov Space and Givens Rotations are used to provide the residual 
norm each iteration
* If complex dielectric, the matvec calls the tree code
separately with the Real part and Imaginary part of the solution and
multiplies by complex constant afterwards. This prevents to modify the 
treecode to accept complex numbers.
* Add to RHS functions the corresponding terms to solve lspr problems.


### Fixed

* Py3 syntax for generator iteration
* Switched from `blas.rotg` to `lapack.lartg` since that apparently works correctly

### Removed

* All commented out code
* Unused imports and unused variables removed.


## 0.2.1
---
### Added

* All documentation is available on github pages
* Support for Cuda 7.5
* One liner for setting up Py2.7 environment (sans PyCUDA)
* Use Doctr to automatically generate documentation using Travis

### Changed
* `config` and `param` files are now globbed for so they can have a name different
  than the folder which contains them.  
* Updated license with new contributors
* Layout of sphinx documentation toolbars (home button added, more verbose layout)

### Fixed

* Uncaught exception when pygbe doesn't run correctly
* Wrong error type in regression test master script
* Performance runs work on non-X backends
* Uncaught out-of-memory exceptions in regression tests

### Removed


## 0.2
---
### Added
* `setup.py` installer
* `argparse` ArgumentParser to handle command line arguments (all optional)
  * `-c` to specify config file
  * `-p` to specify param file
  * `-o` to specify output folder
  * `-g` to specify geometry folder
* Docstrings to all functions
* Checks for NVCC version and to warn if user doesn't have NVCC on PATH
* Sphinx documentation
* In addition to text output, numerical results are stored to a pickled dictionary for easy access
  
### Changed
* Repo structure altered to match Python packaging guidelines.
* Modularized code and removed all relative imports
* All `import *` (excepting files in `scripts/`) have been removed and changed to explicit imports
* Problems are now grouped-by-folder.  A given problem will have the format:
```
lys 
  ˫ lys.param
  ˫ lys.config
  ˫ built_parse.pqr
  ˫ geometry/Lys1.face
  ˫ geometry/Lys1.vert
  ˫ output/

* Support running in current directory by passing '.' as path
```
* Refactored regression tests, added simple caching to avoid test repeats
* Move many, many functions around so that individual `.py` filenames are more descriptive and accurate

### Removed
* Makefiles (functionality replaced by `setup.py`)
* `pygbe_matrix` and `scripts` folder -- to be relocated to a more appropriate repo somewhere
