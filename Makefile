all:
	dune build

clean:
	dune clean


install:
	dune build @install
	dune install

uninstall:
	dune uninstall

test:
	dune exec Samples/src/DeviceQuery/DeviceQuery.exe
	# dune exec Samples/src/VecAdd/VecAdd.exe
	# dune exec Samples/src/Mandelbrot/Mandelbrot.exe

test_sarek:
	dune exec SpocLibs/Benchmarks/Pi/Pi.exe
	dune exec SpocLibs/Benchmarks/Mandelbrot_Sarek/Mandelbrot.exe


check: all install install_sarek samples test #sarek_samples
