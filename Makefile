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

# Core PPX/unit/comparison/e2e builds (GPU execution not required)
test:
	dune build @SpocLibs/Sarek_test/runtest

# Build and run selected GPU e2e tests (requires a working OpenCL/CUDA device)
test_e2e_gpu:
	dune build \
		SpocLibs/Sarek_test/e2e/test_nbody_ppx.exe \
		SpocLibs/Sarek_test/e2e/test_ray_ppx.exe
	dune exec SpocLibs/Sarek_test/e2e/test_nbody_ppx.exe -- --device 0
	dune exec SpocLibs/Sarek_test/e2e/test_ray_ppx.exe -- --device 0

# Force rebuild/re-run of PPX tests even if previously built
test-force:
	dune clean || true
	dune build --force @SpocLibs/Sarek_test/runtest

# Optional CUDA samples (require CUDA toolchain and graphics libs)
test_samples_cuda:
	@if [ -z "$$CUDA_PATH" ] || [ ! -f "$$CUDA_PATH/lib64/libnvrtc.so" ]; then \
	  echo "Skipping CUDA samples (CUDA_PATH/libnvrtc.so not found)"; \
	else \
	  dune build --profile=cuda \
	    Samples/src/DeviceQuery/DeviceQuery.exe \
	    Samples/src/VecAdd/VecAdd.exe \
	    Samples/src/Mandelbrot/Mandelbrot.exe && \
	  dune exec --profile=cuda Samples/src/DeviceQuery/DeviceQuery.exe && \
	  dune exec --profile=cuda Samples/src/VecAdd/VecAdd.exe && \
	  dune exec --profile=cuda Samples/src/Mandelbrot/Mandelbrot.exe ; \
	fi

test_ppx:
	# Build unit/comparison tests and all Sarek PPX e2e binaries (execution may require GPU)
	SKIP_OCAMLFORMAT=1 dune build @SpocLibs/Sarek_test/runtest

# Negative tests - verify that expected compile errors are raised
# Uses --profile=negative to enable the negative test libraries
test_negative:
	@echo "=== Negative tests (expected compile errors) ==="
	@echo "Testing type mismatch detection..."
	@dune build --profile=negative SpocLibs/Sarek_test/negative/neg_test_convention_kernel_fail2.cma 2>&1 | tee /tmp/neg1.out | grep -q "Cannot unify types" && echo "  PASS: type mismatch" || (cat /tmp/neg1.out; false)
	@echo "Testing barrier in diverged control flow..."
	@dune build --profile=negative SpocLibs/Sarek_test/negative/neg_test_barrier_diverged.cma 2>&1 | tee /tmp/neg2.out | grep -q "Barrier called in diverged control flow" && echo "  PASS: barrier diverged" || (cat /tmp/neg2.out; false)
	@echo "Testing superstep in diverged control flow..."
	@dune build --profile=negative SpocLibs/Sarek_test/negative/neg_test_superstep_diverged.cma 2>&1 | tee /tmp/neg3.out | grep -q "Barrier called in diverged control flow" && echo "  PASS: superstep diverged" || (cat /tmp/neg3.out; false)
	@echo "Testing unbound function detection..."
	@dune build --profile=negative SpocLibs/Sarek_test/negative/neg_test_unbound_function.cma 2>&1 | tee /tmp/neg4.out | grep -q "Unbound" && echo "  PASS: unbound function" || (cat /tmp/neg4.out; false)
	@echo "All negative tests passed"

# Run all tests: unit tests, e2e tests, and negative tests
test-all: test test_negative
	@echo "=== All tests passed ==="

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
