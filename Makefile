nprocs+=$(shell getconf _NPROCESSORS_ONLN)

all:
	cd Spoc && $(MAKE) -j$(nprocs) && cd ..

clean:
	cd Spoc && $(MAKE) clean && cd ..
	cd Samples && $(MAKE) clean && cd ..
	cd SpocLibs/Sarek && $(MAKE) clean && cd ../..


install: 
	cd Spoc && $(MAKE) -j$(nprocs) install && cd ..

uninstall:
	cd Spoc && $(MAKE) uninstall && cd ..

samples: install
	cd Samples; $(MAKE)

install_sarek:
	cd SpocLibs/Sarek; $(MAKE) -j$(nprocs) install

sarek_samples:
	cd SpocLibs/Samples/Mandelbrot; $(MAKE)
	cd SpocLibs/Samples/Bitonic_sort; $(MAKE)


check: all install install_sarek samples #sarek_samples
