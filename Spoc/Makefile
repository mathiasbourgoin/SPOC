MLSRC=$(shell ocamldep -sort *.ml)
#Devices.ml Vector.ml OpenCL.ml Cuda.ml Mem.ml Tools.ml KernelOpenCL.ml KernelCuda.ml Kernel.ml
mkfile_path=$(abspath $(lastword $(MAKEFILE_LIST)))
PROJ=$(notdir $(patsubst %/,%,$(dir $(mkfile_path))))
LIBNAME:=$(shell echo $(PROJ) | tr A-Z a-z)

#CSRC=Spoc_c.c  Spoc_cu.c cuda_drvapi_dynlink.c  Custom.c  Kernel_cuda.c  Kernel_opencl.c  Mem_c.c  Opencl_dynlink.c
CSRC=$(wildcard *.c)
COBJ=$(CSRC:.c=.o)
MLOBJ_B=$(MLSRC:.ml=.cmo)
MLOBJ_N=$(MLSRC:.ml=.cmx)
JSSRC=$(filter-out Js/spoc_lib.js, $(wildcard Js/*.js))
CC=ocamlc


all : start | build extension/kernels_ext.cma js


start :
	@echo "\033[43m\033[30mBuilding $(PROJ)\033[0m"

js : Js/$(LIBNAME)_lib.js

Js/$(LIBNAME)_lib.js : $(JSSRC)
	@echo "\033[43m\033[30mBuilding Web$(PROJ)\033[0m"
	@echo "\033[36m[$(LIBNAME)_lib.js]\033[0m"
	@cd Js @@ make


extension/kernels_ext.cma:extension/kernels_ext.ml
	@echo "\033[43m\033[30mBuilding $(PROJ) external kernel extension\033[0m"
	@echo "\033[31m[kernels_ext.cma]\033[0m"
	@cd extension && make 


%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlc -g -for-pack $(PROJ) -c $< 2>>log

%.cmx:%.ml
	@echo  "\033[34m[$@]\033[0m" 
	@ocamlopt -g -for-pack $(PROJ) -c $< 2>>log



%.cmi:%.mli
	@echo  "\033[35m[$@]\033[0m" 
	@ocamlc -for-pack $(PROJ) -c $< 2>>log

%.o:%.c
	@echo  "\033[33m[$@]\033[0m" 
	@$(CC) -g -c $< 2>>log

build : $(LIBNAME).cma $(LIBNAME)level $(LIBNAME).cmxa $(LIBNAME).o

$(LIBNAME).cmo : $(MLOBJ_B)  $(COBJ)
	@echo  "\033[32m[spoc.cmo]\033[0m"
	@ocamlc -g -pack  $(MLOBJ_B) $(COBJ) -o $(LIBNAME).cmo

$(LIBNAME).o : $(COBJ)
	@echo  "\033[33m[$@]\033[0m" 
	@$(CC) -g $(COBJ) -o $(LIBNAME).o


$(LIBNAME).cma: $(LIBNAME).o $(LIBNAME).cmo $(COBJ)
	@echo  "\033[31m[$(LIBNAME).cma]\033[0m"
	@ocamlmklib  -linkall -o $(LIBNAME) $(LIBNAME).cmo $(COBJ) -cclib -lcamlrun

$(LIBNAME).cmx : $(LIBNAME).o $(MLOBJ_N)  $(COBJ)
	@echo  "\033[34m[$(LIBNAME).cmx]\033[0m"	
	@ocamlopt -g -pack $(MLOBJ_N) $(COBJ) -o $(LIBNAME).cmx

$(LIBNAME).cmxa : $(LIBNAME).cmx $(LIBNAME).o $(COBJ)
	@echo  "\033[31m[$(LIBNAME).cmxa]\033[0m"
	@ocamlmklib -linkall -o $(LIBNAME) $(LIBNAME).cmx $(COBJ) -cclib -lasmrun

$(LIBNAME)level : $(LIBNAME).cma extension/kernels_ext.cma
	@echo   "\033[36m[$(LIBNAME)level]\033[0m"
	ocamlmktop -custom -I +camlp4 -cclib -L. dynlink.cma \
	camlp4o.cma extension/kernels_ext.cma str.cma bigarray.cma \
	-cclib -lcamlrun -cclib -lbigarray -cclib -l$(LIBNAME) $(LIBNAME).cma \
	-o $(LIBNAME)level

ocp :
	ocp-build -init
	ocp-build 
	cd extension && make
	cd Js && make


doc : uninstall install
	mkdir -p tmp
	for i in *.mli; do \
	echo $$i; \
	echo "open $(PROJ)" > tmp/$$i; \
	cat $$i >> tmp/$$i; \
	done
	mkdir -p docs
	ocamlfind ocamldoc -thread  -html -package $(LIBNAME) -d docs  tmp/*.mli
	rm -rf tmp

install : all uninstall
	@echo "\033[43m\033[30mInstalling $(PROJ)\033[0m"
	@ocamlfind install $(LIBNAME) *.cma *.a *.so *.cmxa $(LIBNAME).cmi META 2>> log
	@cd extension && make install
	@cd Js && make install

ocp-install:
	ocp-build install
	cp _obuild/$(LIBNAME)/*.cmi `ocamlfind query spoc`
	cd extension && make install
	cd Js && make install

uninstall :
	@echo "\033[43m\033[30mRemoving Spoc\033[0m"
	@cd extension && make uninstall
	@cd Js && make uninstall
	@(ocamlfind remove $(LIBNAME) 2>>log) | true;



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
	ocamldep *.ml *.mli  > .depend


include  .depend

ocp-clean:
	ocp-build clean

clean :
	cd extension && make clean
	cd Js && make clean
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so log a.out $(LIBNAME)level *~ *.a
