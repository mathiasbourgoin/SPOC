all:
	cd Spoc && make && cd ..

clean:
	cd Spoc && make clean && cd ..


install:
	cd Spoc && make install && cd ..

uninstall:
	cd Spoc && make uninstall && cd ..

samples: install
	cd Samples; make

install_sarek:
	cd SpocLibs/Sarek; make && make install

sarek_samples:
	cd SpocLibs/Samples/Mandelbrot; make
	cd SpocLibs/Samples/Bitonic_sort; make


check: all samples install_sarek sarek_samples
