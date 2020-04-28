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
#CC=ocamlfind ocamlc -ccopt "-D SPOC_PROFILE -g -rdynamic"
#OCAMLC=ocamlfind ocamlc  -pp "cppo -q -D SPOC_PROFILE"
#OCAMLOPT=ocamlfind ocamlopt  -pp "cppo -q -D SPOC_PROFILE" -ccopt "-rdynamic -g"
#OCAMLDEP=ocamldep -pp "cppo -q -D SPOC_PROFILE"
CC=ocamlfind ocamlc  -ccopt "-lnvrtc"  -ccopt "-std=gnu99"
OCAMLC=ocamlfind ocamlc -pp "cppo -q" -I .
OCAMLOPT=ocamlfind ocamlopt  -pp "cppo -q" -ccopt "-rdynamic -g"  -I .
OCAMLDEP=ocamldep -pp "cppo -q"
ECHO=printf

all : start | build extension/kernels_ext.cma js


start :
	@$(ECHO) "\033[43m\033[30mBuilding $(PROJ)\033[0m\n"

js : Js/$(LIBNAME)_lib.js

Js/$(LIBNAME)_lib.js : $(JSSRC)
	@$(ECHO) "\033[43m\033[30mBuilding Web$(PROJ)\033[0m\n"
	@$(ECHO) "\033[36m[$(LIBNAME)_lib.js]\033[0m\n"
	@cd Js && $(MAKE)


extension/kernels_ext.cma:extension/kernels_ext.ml
	@$(ECHO) "\033[43m\033[30mBuilding $(PROJ) external kernel extension\033[0m\n"
	@$(ECHO) "\033[31m[kernels_ext.cma]\033[0m\n"
	@cd extension && $(MAKE)


%.cmo:%.ml %.cmi
	@$(ECHO) "\033[32m[$@]\033[0m\n"
	@$(OCAMLC) -I . -g -for-pack $(PROJ) -c  $<

%.cmx:%.ml %.cmi
	@$(ECHO) "\033[34m[$@]\033[0m\n"
	@$(OCAMLOPT) -I . -g -for-pack $(PROJ) -c  $<


%.cmi:%.mli
	@$(ECHO)  "\033[35m[$@]\033[0m\n"
	@$(OCAMLC) -for-pack $(PROJ) -c  $< #2>>log

%.o:%.c
	@$(ECHO)  "\033[33m[$@]\033[0m\n"
	$(CC) -g -c $<

build : $(LIBNAME).cma $(LIBNAME)level $(LIBNAME).cmxa $(LIBNAME).o

$(LIBNAME).cmo : $(MLOBJ_B)  $(COBJ)
	@$(ECHO)  "\033[32m[spoc.cmo]\033[0m\n"
	$(OCAMLC) -g -pack  $(MLOBJ_B) $(COBJ) -o $(LIBNAME).cmo

$(LIBNAME).o : $(COBJ)
	@$(ECHO)  "\033[33m[$@]\033[0m\n"
	@$(CC) -g $(COBJ) -o $(LIBNAME).o


$(LIBNAME).cma: $(LIBNAME).o $(LIBNAME).cmo $(COBJ)
	@$(ECHO)  "\033[31m[$(LIBNAME).cma]\033[0m\n"
	@ocamlmklib  -linkall -o $(LIBNAME) $(LIBNAME).cmo $(COBJ) -cclib -lcamlrun

$(LIBNAME).cmx : $(LIBNAME).o $(MLOBJ_N)  $(COBJ)
	@$(ECHO)  "\033[34m[$(LIBNAME).cmx]\033[0m\n"
	@$(OCAMLOPT) -g -pack $(MLOBJ_N) $(COBJ) -o $(LIBNAME).cmx

$(LIBNAME).cmxa : $(LIBNAME).cmx $(LIBNAME).o $(COBJ)
	@$(ECHO)  "\033[31m[$(LIBNAME).cmxa]\033[0m\n"
	@ocamlmklib -linkall -o $(LIBNAME) $(LIBNAME).cmx $(COBJ) -cclib -lasmrun

$(LIBNAME)level : $(LIBNAME).cma extension/kernels_ext.cma
	@$(ECHO)   "\033[36m[$(LIBNAME)level]\033[0m\n"
	ocamlmktop -custom -thread -I +camlp4 -cclib -L. dynlink.cma	\
	-cclib -lthreads camlp4o.cma extension/kernels_ext.cma		\
	str.cma bigarray.cma -cclib -lcamlrun 	\
	-cclib -lnvrtc -ccopt -L$(CUDA_PATH)/lib64 	\
	-cclib -lunix unix.cma -cclib -l$(LIBNAME) $(LIBNAME).cma -o	\
	$(LIBNAME)level

doc : uninstall install
	mkdir -p tmp
	for i in *.mli; do \
	$(ECHO) $$i; \
	$(ECHO) "open $(PROJ)" > tmp/$$i; \
	cat $$i >> tmp/$$i; \
	done
	mkdir -p docs
	ocamlfind ocamldoc -thread  -html -package $(LIBNAME) -d docs  tmp/*.mli
	rm -rf tmp

install : all uninstall
	@$(ECHO) "\033[43m\033[30mInstalling $(PROJ)\033[0m\n"
	@ocamlfind install $(LIBNAME) *.cma *.a *.so *.cmxa $(LIBNAME).cmi META #2>> log
	@cd extension && $(MAKE) install
	@cd Js && $(MAKE) install

uninstall :
	@$(ECHO) "\033[43m\033[30mRemoving Spoc\033[0m\n"
	@cd extension && $(MAKE) uninstall
	@cd Js && $(MAKE) uninstall
	@(ocamlfind remove $(LIBNAME) 2>>log) | true;

depend :
	$(OCAMLDEP) *.ml *.mli  > .depend


include  .depend

clean :
	cd extension && $(MAKE) clean
	cd Js && $(MAKE) clean
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so log a.out $(LIBNAME)level *~ *.a
