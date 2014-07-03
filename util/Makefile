SWIG = swig
SWIG_OPTS = -c++ -python

CC = g++ 
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

PY_VER=$(shell python -c "import sys;t='{v[0]}.{v[1]}'.format(v=list(sys.version_info[:2]));sys.stdout.write(t)")
PY_INC=/usr/include/python$(PY_VER)
PY_LIB=/usr/include/python$(PY_VER)

all: swig link

swig: semi_analyticalwrap.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c semi_analyticalwrap.cpp semi_analyticalwrap_wrap.cxx -I $(PY_INC) -I $(PY_LIB)
	$(CC) $(OPTS) -shared $? -o _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o

clean:
	rm -f _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o semi_analyticalwrap_wrap.cxx semi_analyticalwrap.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
