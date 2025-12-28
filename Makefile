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

# Run interpreter tests (no GPU required)
test_interpreter:
	@echo "=== Interpreter unit tests ==="
	dune build SpocLibs/Sarek_test/unit/test_interp.exe
	@echo "Running interpreter unit tests..."
	dune exec SpocLibs/Sarek_test/unit/test_interp.exe
	@echo "=== Interpreter e2e tests ==="
	dune build SpocLibs/Sarek_test/e2e/test_vector_add.exe
	@echo "Running vector_add on interpreter..."
	dune exec SpocLibs/Sarek_test/e2e/test_vector_add.exe -- --interpreter -s 64 -b 16
	@echo "=== Interpreter tests passed ==="

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

# Run new comprehensive Sarek e2e tests (GPU required)
test_comprehensive:
	@echo "=== Comprehensive Sarek e2e tests ==="
	dune build \
		SpocLibs/Sarek_test/e2e/test_stencil.exe \
		SpocLibs/Sarek_test/e2e/test_matrix_mul.exe \
		SpocLibs/Sarek_test/e2e/test_reduce.exe \
		SpocLibs/Sarek_test/e2e/test_histogram.exe \
		SpocLibs/Sarek_test/e2e/test_complex_types.exe \
		SpocLibs/Sarek_test/e2e/test_math_intrinsics.exe \
		SpocLibs/Sarek_test/e2e/test_bitwise_ops.exe \
		SpocLibs/Sarek_test/e2e/test_scan.exe \
		SpocLibs/Sarek_test/e2e/test_transpose.exe \
		SpocLibs/Sarek_test/e2e/test_sort.exe \
		SpocLibs/Sarek_test/e2e/test_convolution.exe \
		SpocLibs/Sarek_test/e2e/test_mandelbrot.exe
	@echo "Running on native parallel CPU..."
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_stencil.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_matrix_mul.exe -- --native-parallel -s 1024
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_reduce.exe -- --native-parallel -s 8192
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_histogram.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_complex_types.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_math_intrinsics.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_bitwise_ops.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_scan.exe -- --native-parallel -s 256
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_transpose.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_sort.exe -- --native-parallel -s 512
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_convolution.exe -- --native-parallel -s 4096
	LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_mandelbrot.exe -- --native-parallel -s 4096
	@echo "=== Comprehensive e2e tests passed ==="

# Run all tests: unit tests, e2e tests, and negative tests
test-all: test test_interpreter test_negative
	@echo "=== All tests passed ==="

# E2E tests - quick verification with small datasets comparing GPU vs native CPU
E2E_TESTS = test_stencil test_matrix_mul test_reduce test_histogram \
            test_complex_types test_math_intrinsics test_bitwise_ops \
            test_scan test_transpose test_sort test_convolution test_mandelbrot

test-e2e:
	@echo "=== E2E Tests (small datasets, verification enabled) ==="
	@dune build $(addprefix SpocLibs/Sarek_test/e2e/,$(addsuffix .exe,$(E2E_TESTS)))
	@echo ""
	@echo "--- Stencil ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_stencil.exe -- -s 1024
	@echo ""
	@echo "--- Matrix Multiplication ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_matrix_mul.exe -- -s 256
	@echo ""
	@echo "--- Reduction ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_reduce.exe -- -s 2048
	@echo ""
	@echo "--- Histogram ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_histogram.exe -- -s 1024
	@echo ""
	@echo "--- Complex Types ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_complex_types.exe -- -s 1024
	@echo ""
	@echo "--- Math Intrinsics ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_math_intrinsics.exe -- -s 1024
	@echo ""
	@echo "--- Bitwise Operations ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_bitwise_ops.exe -- -s 1024
	@echo ""
	@echo "--- Scan ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_scan.exe -- -s 256
	@echo ""
	@echo "--- Transpose ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_transpose.exe -- -s 1024
	@echo ""
	@echo "--- Sort ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_sort.exe -- -s 256
	@echo ""
	@echo "--- Convolution ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_convolution.exe -- -s 1024
	@echo ""
	@echo "--- Mandelbrot ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_mandelbrot.exe -- -s 1024
	@echo ""
	@echo "=== All E2E tests passed ==="

# Benchmarks - run all tests with --benchmark to compare all devices
benchmarks:
	@echo "=============================================="
	@echo "       SAREK BENCHMARK SUITE"
	@echo "=============================================="
	@echo ""
	@dune build $(addprefix SpocLibs/Sarek_test/e2e/,$(addsuffix .exe,$(E2E_TESTS)))
	@echo ""
	@echo "--- Stencil (1D/2D with shared memory) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_stencil.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Matrix Multiplication (naive + tiled) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_matrix_mul.exe -- --benchmark -s 4096
	@echo ""
	@echo "--- Reduction (sum, max, dot product) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_reduce.exe -- --benchmark -s 131072
	@echo ""
	@echo "--- Histogram ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_histogram.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Complex Types (records, particles) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_complex_types.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Math Intrinsics (sin, cos, exp, log, sqrt) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_math_intrinsics.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Bitwise Operations ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_bitwise_ops.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Scan (prefix sum) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_scan.exe -- --benchmark -s 256
	@echo ""
	@echo "--- Transpose (naive + coalesced) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_transpose.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Sort (bitonic, odd-even) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_sort.exe -- --benchmark -s 4096
	@echo ""
	@echo "--- Convolution (1D, 2D, Sobel) ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_convolution.exe -- --benchmark -s 65536
	@echo ""
	@echo "--- Mandelbrot / Julia ---"
	@LD_LIBRARY_PATH=/opt/cuda/lib64:$$LD_LIBRARY_PATH dune exec SpocLibs/Sarek_test/e2e/test_mandelbrot.exe -- --benchmark -s 65536
	@echo ""
	@echo "=============================================="
	@echo "       BENCHMARK COMPLETE"
	@echo "=============================================="

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
