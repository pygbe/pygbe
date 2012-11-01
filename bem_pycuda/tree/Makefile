SWIG = swig
SWIG_OPTS = -c++ -python
CC = g++
OPTS = -fPIC -funroll-loops -O3 -msse3 -fopenmp

all: swig1 swig2 swig3 link

swig1: getCoeffwrap.i
	$(SWIG) $(SWIG_OPTS) $?

swig2: direct_c.i
	$(SWIG) $(SWIG_OPTS) $?

swig3: calculateMultipoles.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c getCoeffwrap.cpp getCoeffwrap_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -fPIC -shared $? -o _getCoeffwrap.so getCoeffwrap.o getCoeffwrap_wrap.o
	$(CC) $(OPTS) -c direct_c.cpp direct_c_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -fPIC -shared $? -o _direct_c.so direct_c.o direct_c_wrap.o
	$(CC) $(OPTS) -c calculateMultipoles.cpp calculateMultipoles_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -fPIC -shared $? -o _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o

clean:
	rm -f _getCoeffwrap.so getCoeffwrap.o getCoeffwrap_wrap.o getCoeffwrap_wrap.cxx getCoeffwrap.py *.pyc
	rm -f _direct_c.so direct_c.o direct_c_wrap.o direct_c_wrap.cxx direct_c.py *.pyc
	rm -f _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o calculateMultipoles_wrap.cxx calculateMultipoles.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
