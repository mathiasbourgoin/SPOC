MLSRC=$(shell ocamldep -sort *.ml)
#Kirc_Ast.ml Kirc_Cuda.ml Kirc_OpenCL.ml Kirc.ml
MLI=$(shellocamldep -sort *.mli)
#Kirc_Ast.mli Kirc.mli
MLOBJ_B= $(MLSRC:.ml=.cmo) 
MLOBJ_N=$(MLSRC:.ml=.cmx)
EXTSRC=$(wildcard extension/*.ml)

all : start | build extension/kernels_int.cma

extension/kernels_int.cma: $(EXTSRC)
	@echo "\033[43m\033[30mBuilding SPOC external kernel extension\033[0m"
	@echo "\033[31m[kernels_ext.cma]\033[0m"
	@cd extension && make 

start :
	@echo "\033[43m\033[30mBuilding Sarek\033[0m"

%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc -c $< 

%.cmx:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlopt -thread -package spoc -c $< 

%.cmi:%.mli
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc -c $< 


sarek.cma: $(MLOBJ_B)
	@echo  "\033[31m[sarek.cma]\033[0m"
	ocamlfind ocamlc -thread -package spoc -a -o sarek.cma $(MLOBJ_B)

libsarek.a:sarek.cma
	@cp sarek.a libsarek.a	

spoclevel-sarek : sarek.cmxa sarek.cma extension/kernels_int.cma
	@echo   "\033[36m[spoclevel-sarek]\033[0m"
	@ocamlmktop -custom -I +camlp4 -cclib -L. dynlink.cma \
	-cclib -L`ocamlfind query spoc` -I `ocamlfind query spoc` \
	camlp4o.cma extension/kernels_int.cma str.cma bigarray.cma \
	-cclib -lcamlrun -cclib -lbigarray \
	spoc.cma -cclib -lspoc \
	-cclib -lsarek sarek.cma \
	-o spoclevel-sarek 

#sarek.cma: sarek.cmo
#	@echo  "\033[31m[sarek.cma]\033[0m"
#	@ocamlmklib  -linkall -o sarek sarek.cmo -cclib -lcamlrun

sarek.cmxa : $(MLOBJ_N)
	@echo  "\033[31m[sarek.cmxa]\033[0m"
	@ocamlfind ocamlopt -thread -package spoc -a  $(MLOBJ_N) -o sarek.cmxa
#@ocamlmklib -linkall -o sarek $(MLOBJ_N)  -cclib -lasmrun

build: sarek.cmxa sarek.cma #spoclevel-sarek

ocp:
	ocp-build -init
	ocp-build
	cd extension && make

depend :
	ocamldep *.ml *.mli > .depend

include  .depend


ocp-clean :
	ocp-build clean	

clean :
	cd extension && make clean
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so log a.out spoclevel-sarek *~ sarek.a


install : all uninstall
	@echo "\033[43m\033[30mInstalling Sarek\033[0m"
	@ocamlfind install sarek *.cma *.a  *.cmxa *.cmi META 2>> log
	@cd extension && make install

ocp-install:
	ocp-build install
	cp _obuild/sarek/*.cm* `ocamlfind query sarek`
	cd extension && make install

uninstall :
	(ocamlfind query sarek && ocamlfind remove sarek) || echo "sarek not installed"
	cd extension && make uninstall
