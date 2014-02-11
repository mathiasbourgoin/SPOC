MLSRCDIRS = CustomPi DeviceQuery Game \
			Mandelbrot Mandelbrot2 VecAdd \
			VecAdd_MultiGPU 

all: 
	for i in $(MLSRCDIRS); do \
	( cd $$i; make; ) \
	done

clean:
	for i in $(MLSRCDIRS); do \
	( cd $$i; make clean; ) \
	done
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm *.tmp *.tmp.dll

