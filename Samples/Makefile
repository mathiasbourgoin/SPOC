MLSRCDIRS = CustomPi DeviceQuery VecAdd VecAdd_MultiGPU 

MLSRCDIRSGR = Game Mandelbrot Mandelbrot2 

KERNELDIR = ./kernels
BINDIR=./build/

all:  
	cd src && make && cd ..
	mkdir -p $(KERNELDIR)
	rm -rf $(BINDIR)
	mkdir -p $(BINDIR)/Native/$(KERNELDIR)
	mkdir -p $(BINDIR)/Bytecode/$(KERNELDIR)
	for i in $(MLSRCDIRS); do \
		( cp src/$$i/*.byte* $(BINDIR)/Bytecode; cp src/$$i/*.asm* $(BINDIR)/Native; \
		if [ -e src/$$i/kernels ]; then \
			rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Bytecode/;\
	 		rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Native/;\
		fi;)\
	done
	if (ocamlfind query graphics 1>/dev/null 2>&1); then \
		for i in $(MLSRCDIRSGR); do \
			( cp src/$$i/*.byte* $(BINDIR)/Bytecode; cp src/$$i/*.asm* $(BINDIR)/Native; \
			if [ -e src/$$i/kernels ]; then \
				rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Bytecode/;\
	 		rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Native/;\
			fi;)\
		done;\
	fi	
	cp ../Spoc/*.ptx $(KERNELDIR)
	cp ../Spoc/*.cl $(KERNELDIR)
	cp -a $(KERNELDIR) $(BINDIR)/Bytecode/
	cp -a $(KERNELDIR) $(BINDIR)/Native/

../Spoc/Spoc_kernels.ptx:
	cd ../Spoc
	make clean
	make

%.nat:
	cd src && make 

%.byte:
	cd src && make 

%.exe: 
	cd src && make

clean:
	cd src && make clean
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm *.tmp *.tmp.dll \#*\#

mrproper : clean
	rm -rf $(BINDIR)

