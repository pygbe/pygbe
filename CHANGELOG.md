# PyGBe Change Log
----

## Current developments
---
### Added
* `setup.py` installer

### Changed
* Repo structure altered to match Python packaging guidelines.
* Modularized code and removed all relative imports
* All `import *` (excepting files in `scripts/`) have been removed and changed to explicit imports

### Removed
* Makefiles (functionality replaced by `setup.py`)
