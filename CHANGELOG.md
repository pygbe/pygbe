# PyGBe Change Log
----

## Current developments
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
```
* Refactored regression tests, added simple caching to avoid test repeats
* Move many, many functions around so that individual `.py` filenames are more descriptive and accurate


### Removed
* Makefiles (functionality replaced by `setup.py`)
