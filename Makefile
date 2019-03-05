nprocs+=$(shell getconf _NPROCESSORS_ONLN)

all:
	cd Spoc && $(MAKE) -j$(nprocs) && cd ..

clean:
	cd Spoc && $(MAKE) clean && cd ..


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


test:
	cd Samples ; $(MAKE) --no-print-directory test | tee "/tmp/log_spoc_test_samples"
	@if grep "KO" "/tmp/log_spoc_test_samples" ; \
	then printf "\e[1mALL TESTS: \033[0;31mKO\033[0m\n" ; exit 1 ; \
	else printf "\e[1mALL TESTS: \033[0;32mOK\033[0m\n" ; \
	fi 


check: all install install_sarek samples test #sarek_samples
