SWIG = swig
SWIG_OPTS = -c++ -python
CC = g++
OPTS = -fPIC -O3 -funroll-loops -msse3 -fopenmp

all: calc_mult mult dir link

calc_mult: calculateMultipoles.i
	$(SWIG) $(SWIG_OPTS) $?

mult: multipole.i
	$(SWIG) $(SWIG_OPTS) $?

dir: direct.i
	$(SWIG) $(SWIG_OPTS) $?

link:  
	$(CC) $(OPTS) -c calculateMultipoles.cpp calculateMultipoles_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -shared $? -o _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o
	$(CC) $(OPTS) -c multipole.cpp multipole_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -shared $? -o _multipole.so multipole.o multipole_wrap.o
	$(CC) $(OPTS) -c direct.cpp direct_wrap.cxx -I/usr/include/python2.6 -I/usr/lib/python2.6
	$(CC) $(OPTS) -shared $? -o _direct.so direct.o direct_wrap.o

clean:
	rm -f _calculateMultipoles.so calculateMultipoles.o calculateMultipoles_wrap.o calculateMultipoles_wrap.cxx calculateMultipoles.py *.pyc
	rm -f _multipole.so multipole.o multipole_wrap.o multipole_wrap.cxx multipole.py *.pyc
	rm -f _direct.so direct.o direct_wrap.o direct_wrap.cxx direct.py *.pyc

.cxx.o:
	$(CC) $(OPTS) -c $< -o $@

.cpp.o:
	$(CC) $(OPTS) -c $< -o $@
