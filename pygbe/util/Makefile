SWIG = swig
SWIG_OPTS = -c++ -python

CC = g++ 
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

PY_BASE_DIR=$(shell python -c "from distutils.spawn import find_executable;import sys;t=find_executable('python')[:-11];sys.stdout.write(t)")
PY_VER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY_INC=$(PY_BASE_DIR)/include/python$(PY_VER)m
PY_LIB=$(PY_BASE_DIR)/include/python$(PY_VER)
NUMPY_LIB=$(PY_BASE_DIR)/lib/python$(PY_VER)/site-packages/numpy/core/include

all: swig link

swig: semi_analyticalwrap.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c semi_analyticalwrap.cpp semi_analyticalwrap_wrap.cxx -I $(PY_INC) -I $(PY_LIB) -I $(NUMPY_LIB)
	$(CC) $(OPTS) -shared $? -o _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o

clean:
	rm -f _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o semi_analyticalwrap_wrap.cxx semi_analyticalwrap.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
