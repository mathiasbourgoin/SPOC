nprocs=$(shell getconf _NPROCESSORS_ONLN)

all:
	cd Spoc && make -j$(nprocs) && cd ..

clean:
	cd Spoc && make clean && cd ..


install: 
	cd Spoc && make -j$(nprocs) install && cd ..

uninstall:
	cd Spoc && make uninstall && cd ..

samples: install
	cd Samples; make

install_sarek: 
	cd SpocLibs/Sarek; make -j$(nprocs) install

sarek_samples:
	cd SpocLibs/Samples/Mandelbrot; make
	cd SpocLibs/Samples/Bitonic_sort; make


check:
	make all install
	make install_sarek
	make samples #sarek_samples
