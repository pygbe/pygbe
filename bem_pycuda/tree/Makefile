SWIG = swig
SWIG_OPTS = -c++ -python
CC = g++
OPTS = -O3 -funroll-loops -msse3 -fopenmp -fPIC

all: swig link

swig: getCoeffwrap.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c getCoeffwrap.cpp getCoeffwrap_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	g++ -fopenmp -fPIC -shared $? -o _getCoeffwrap.so getCoeffwrap.o getCoeffwrap_wrap.o

clean:
	rm -f _getCoeffwrap.so getCoeffwrap.o getCoeffwrap_wrap.o getCoeffwrap_wrap.cxx getCoeffwrap.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
