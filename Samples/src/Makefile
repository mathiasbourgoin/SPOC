MLSRCDIRS = CustomPi DeviceQuery VecAdd VecAdd_MultiGPU

MLSRCDIRSGR = Game Mandelbrot Mandelbrot2

all:
	for i in $(MLSRCDIRS); do \
		( cd $$i; $(MAKE); ) \
	done
	if (ocamlfind query graphics 1>/dev/null 2>&1); then \
		for i in  $(MLSRCDIRSGR); do \
			( cd $$i; $(MAKE); ) \
		done; \
	fi

test:
	for i in $(MLSRCDIRS); do \
		( cd $$i; $(MAKE) test; ) \
	done


clean:
	for i in $(MLSRCDIRS) $(MLSRCDIRSGR); do \
	( cd $$i; $(MAKE) clean; ) \
	done
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm *.tmp *.tmp.dll

