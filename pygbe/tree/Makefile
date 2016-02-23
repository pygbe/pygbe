SWIG = swig
SWIG_OPTS = -c++ -python

CC = g++
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

PY_BASE_DIR=$(shell python -c "from distutils.spawn import find_executable;import sys;t=find_executable('python')[:-11];sys.stdout.write(t)")
PY_VER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY_INC=$(PY_BASE_DIR)/include/python$(PY_VER)m
PY_LIB=$(PY_BASE_DIR)/include/python$(PY_VER)
NUMPY_LIB=$(PY_BASE_DIR)/lib/python$(PY_VER)/site-packages/numpy/core/include

all: calc_mult mult dir link

calc_mult: calculateMultipoles.i
	$(SWIG) $(SWIG_OPTS) $?

mult: multipole.i
	$(SWIG) $(SWIG_OPTS) $?

dir: direct.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c calculateMultipoles.cpp calculateMultipoles_wrap.cxx -I $(PY_INC) -I $(PY_LIB) -I $(NUMPY_LIB)
	$(CC) $(OPTS) -shared $? -o _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o
	$(CC) $(OPTS) -c multipole.cpp multipole_wrap.cxx -I $(PY_INC) -I $(PY_LIB) -I $(NUMPY_LIB)
	$(CC) $(OPTS) -shared $? -o _multipole.so multipole.o multipole_wrap.o
	$(CC) $(OPTS) -c direct.cpp direct_wrap.cxx -I $(PY_INC) -I $(PY_LIB) -I $(NUMPY_LIB)
	$(CC) $(OPTS) -shared $? -o _direct.so direct.o direct_wrap.o

clean:
	rm -f _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o calculateMultipoles_wrap.cxx calculateMultipoles.py *.pyc
	rm -f _multipole.so multipole.o multipole_wrap.o multipole_wrap.cxx multipole.py *.pyc
	rm -f _direct.so direct.o direct_wrap.o direct_wrap.cxx direct.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
