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
* Forked `which` clone to check if `nvcc` is on PATH and fail gracefully if not.
  

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
### Removed
* Makefiles (functionality replaced by `setup.py`)
