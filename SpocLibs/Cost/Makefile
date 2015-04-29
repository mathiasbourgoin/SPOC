MLSRC=Cost.ml
MLOBJ_B= $(MLSRC:.ml=.cmo) 
MLOBJ_N=$(MLSRC:.ml=.cmx)


all : start | build 

start :
	@echo "\033[43m\033[30mBuilding Cost\033[0m"


%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc,sarek -c $< 2>>log

%.cmx:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlopt -thread -package spoc,sarek -c $< 2>>log

%.cmi:%.mli
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc -c $< 2>>log


build: cost.cma cost.cmxa

cost.cma: $(MLOBJ_B)
	@echo  "\033[31m[$@]\033[0m"
	@ocamlfind ocamlc -thread -package spoc,sarek -a -o cost.cma $(MLOBJ_B)

cost.cmxa: $(MLOBJ_N)
	@echo  "\033[31m[$@]\033[0m"
	@ocamlfind ocamlopt -thread -package spoc,sarek -a -o cost.cmxa $(MLOBJ_N)

test:cost.cma test.ml
	@ocamlfind ocamlopt -thread -package spoc,sarek -linkpkg  -package sarek_syntax  -syntax camlp4o  cost.cmxa -o test test.ml


clean :
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so log a.out spoclevel-sarek *~


