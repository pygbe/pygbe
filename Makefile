all:
	make -C pygbe/tree all
	make -C pygbe/util all
	python setup.py build_ext --inplace
	python setup.py install
cleanall:
	rm -f pygbe/*.pyc
	make -C pygbe/tree clean
	make -C pygbe/util clean
