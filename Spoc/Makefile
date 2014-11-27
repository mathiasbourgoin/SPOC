MLSRC=Devices.ml Vector.ml OpenCL.ml Cuda.ml Mem.ml Tools.ml KernelOpenCL.ml KernelCuda.ml Kernel.ml
CSRC=$(wildcard *.c)
COBJ=$(CSRC:.c=.o)
MLOBJ_B=$(MLSRC:.ml=.cmo)
MLOBJ_N=$(MLSRC:.ml=.cmx)
JSSRC=$(filter-out Js/spoc_lib.js, $(wildcard Js/*.js))
CC=ocamlc


all : start | build extension/kernels_ext.cma js

start :
	@echo "\033[43m\033[30mBuilding SPOC\033[0m"

js : Js/spoc_lib.js

Js/spoc_lib.js : $(JSSRC)
	@echo "\033[43m\033[30mBuilding WebSPOC\033[0m"
	@echo "\033[36m[spoc_lib.js]\033[0m"
	@cd Js @@ make


extension/kernels_ext.cma:extension/kernels_ext.ml
	@echo "\033[43m\033[30mBuilding SPOC external kernel extension\033[0m"
	@echo "\033[31m[kernels_ext.cma]\033[0m"
	@cd extension && make 


%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlc -for-pack Spoc -c $< 2>>log

%.cmx:%.ml
	@echo  "\033[34m[$@]\033[0m" 
	@ocamlopt -for-pack Spoc -c $< 2>>log



%.cmi:%.mli
	@echo  "\033[35m[$@]\033[0m" 
	@ocamlc -for-pack Spoc -c $< 2>>log

%.o:%.c
	@echo  "\033[33m[$@]\033[0m" 
	@$(CC) -c $< 2>>log

build : spoc.cma spoclevel spoc.cmxa spoc.o

spoc.cmo : $(MLOBJ_B)  $(COBJ)
	@echo  "\033[32m[spoc.cmo]\033[0m"
	@ocamlc -pack  $(MLOBJ_B) $(COBJ) -o spoc.cmo

spoc.o : $(COBJ)
	@echo  "\033[33m[$@]\033[0m" 
	@$(CC) $(COBJ) -o spoc.o


spoc.cma: spoc.o spoc.cmo $(COBJ)
	@echo  "\033[31m[spoc.cma]\033[0m"
	@ocamlmklib  -linkall -o spoc spoc.cmo $(COBJ) -cclib -lcamlrun

spoc.cmx : spoc.o $(MLOBJ_N)  $(COBJ)
	@echo  "\033[34m[spoc.cmx]\033[0m"	
	@ocamlopt -pack $(MLOBJ_N) $(COBJ) -o spoc.cmx

spoc.cmxa : spoc.cmx spoc.o $(COBJ)
	@echo  "\033[31m[spoc.cmxa]\033[0m"
	@ocamlmklib -linkall -o spoc spoc.cmx $(COBJ) -cclib -lasmrun

spoclevel : spoc.cma extension/kernels_ext.cma
	@echo   "\033[36m[spoclevel]\033[0m"
	@ocamlmktop -custom -I +camlp4 -cclib -L. dynlink.cma \
	camlp4o.cma extension/kernels_ext.cma str.cma bigarray.cma \
	-cclib -lcamlrun -cclib -lbigarray -cclib -lspoc spoc.cma \
	-o spoclevel

ocp :
	ocp-build -init
	ocp-build 
	cd extension && make
	cd Js && make


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

install : all uninstall
	@echo "\033[43m\033[30mInstalling SPOC\033[0m"
	@ocamlfind install spoc *.cma *.a *.so *.cmxa spoc.cmi META 2>> log
	@cd extension && make install
	@cd Js && make install

ocp-install:
	ocp-build install
	cp _obuild/spoc/*.cmi `ocamlfind query spoc`
	cd extension && make install
	cd Js && make install

uninstall :
	@cd extension && make uninstall
	@cd Js && make uninstall
	@ocamlfind remove spoc 2>>log



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


depend :
	ocamldep *.ml *.mli > .depend

include  .depend

clean :
	ocp-build clean
	cd extension && make clean
	cd Js && make clean
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so
