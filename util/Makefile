SWIG = swig
SWIG_OPTS = -c++ -python
CC = g++
OPTS = -O3 -funroll-loops -msse3 -fopenmp -fPIC

all: swig link

swig: semi_analyticalwrap.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c semi_analyticalwrap.cpp semi_analyticalwrap_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	g++ -fopenmp -fPIC -shared $? -o _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o

clean:
	rm -f _semi_analyticalwrap.so semi_analyticalwrap.o semi_analyticalwrap_wrap.o semi_analyticalwrap_wrap.cxx semi_analyticalwrap.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
