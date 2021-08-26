all:
	dune build

clean:
	dune clean

opam:
	dune build @install
	echo 'available: [ os = "linux" ]'  >> spoc.opam
	echo 'available: [ os = "linux" ]'  >> sarek.opam
	echo 'available: [ os = "linux" ]'  >> spoc_ppx.opam


install: opam
	dune install

uninstall:
	dune uninstall

test:
	dune exec Samples/src/DeviceQuery/DeviceQuery.exe
	dune exec Samples/src/VecAdd/VecAdd.exe
	dune exec Samples/src/Mandelbrot/Mandelbrot.exe

test_sarek:
	echo "Compiling Sarek samples"
	dune build SpocLibs/Benchmarks/Pi/Pi.exe
	dune build SpocLibs/Benchmarks/Mandelbrot_Sarek/Mandelbrot.exe
	dune build SpocLibs/Samples/Bitonic_sort/Bitonic_sort.exe
	echo "Running OpenCL compatible samples"
	dune exec SpocLibs/Samples/Bitonic_sort/Bitonic_sort.exe
	## Cuda Samples cannot be executed on CI
	# dune exec SpocLibs/Benchmarks/Pi/Pi.exe
	# dune exec SpocLibs/Benchmarks/Mandelbrot_Sarek/Mandelbrot.exe

check: all install install_sarek samples test test_sarek

mr_proper: clean
	rm -rf _build

release:
	dune-release tag
	dune-release distrib
	dune-release publish
	dune-release opam pkg
	dune-release opam submit
