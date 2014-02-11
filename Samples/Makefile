MLSRCDIRS = CustomPi DeviceQuery Game Mandelbrot Mandelbrot2 VecAdd \
			VecAdd_MultiGPU 

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
	 rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Bytecode/; rsync -a --exclude .svn src/$$i/kernels $(BINDIR)/Native/; )\
	done
	cp ../Spoc/*.ptx $(KERNELDIR)
	cp ../Spoc/*.cl $(KERNELDIR)
#	rsync -a --exclude .svn  $(KERNELDIR) $(BINDIR)/Bytecode/
#	rsync -a --exclude .svn $(KERNELDIR) $(BINDIR)/Native/
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

