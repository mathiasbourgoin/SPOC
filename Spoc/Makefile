all : build

build :
	ocp-build -init
	ocp-build 
	cd extension && make
	cd Js && make

clean :
	ocp-build clean
	cd extension && make clean
	cd Js && make clean
	rm -rf tmp docs ocp-build.root.* ~*

doc : uninstall install
	mkdir -p tmp
	for i in *.mli; do \
	echo $$i; \
	echo "open Spoc" > tmp/$$i; \
	cat $$i >> tmp/$$i; \
	done
	mkdir -p docs
	ocamlfind ocamldoc -thread  -html -package spoc -d docs  tmp/*.mli
	rm -rf tmp

install : uninstall
	ocp-build install
	cp _obuild/spoc/*.cmi `ocamlfind query spoc`
	cd extension && make install
	cd Js && make install

uninstall :
	ocamlfind remove spoc
	cd extension && make uninstall
	cd Js && make uninstall



android:
	for i in "cuda_drvapi_dynlink.c"  "Custom.c" "Kernel_cuda.c" \
	 	  "Kernel_opencl.c" "Mem_c.c" "Opencl_dynlink.c" \
		  "Spoc_c.c" "Spoc_cu.c" ; do \
	ocamlfind -toolchain android ocamlopt -package bigarray $$i;\
	done
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Devices.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Devices.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Vector.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Vector.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Cuda.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Cuda.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c OpenCL.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c OpenCL.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Mem.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Mem.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Kernel.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Kernel.ml

	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c KernelCuda.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c KernelOpenCL.ml
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Tools.mli
	ocamlfind -toolchain android ocamlopt -for-pack Spoc -package bigarray -linkpkg -c Tools.ml
	ocamlfind -toolchain android ocamlopt -pack -o Spoc.cmx -package bigarray -linkpkg Devices.cmx Vector.cmx Cuda.cmx OpenCL.cmx Mem.cmx Kernel.cmx KernelCuda.cmx KernelOpenCL.cmx Tools.cmx *.o
	ocamlfind -toolchain android ocamlmklib -linkall Spoc.cmx -o spoc_android
	ar rc spoc_android.a *.o
