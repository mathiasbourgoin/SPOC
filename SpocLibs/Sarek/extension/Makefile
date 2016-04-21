P4FLAGS =  -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma 

all:
	@make -j4 kernels_int.cma 


install: kernels_int.cma
	@ocamlfind install sarek_syntax *.cm* META

uninstall :
	(ocamlfind query sarek_syntax && ocamlfind remove sarek_syntax) || echo "sarek_syntax not installed"

kernels_int.cma:debug.cmo sarek_types.cmo typer.cmo gen_caml.cmo gen_kir.cmo mparser.cmo  sarek_grammar.cmo
	@ocamlfind ocamlc -a $^ -o kernels_int.cma


%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc $(P4FLAGS) $^ -c

gen_caml.cmo : sarek_types.cmo gen_caml.ml

gen_kir.cmo : sarek_types.cmo gen_kir.ml


mparser.cmo:sarek_types.cmo mparser.ml

debug.cmo : debug.ml

sarek_types.cmo:debug.cmo sarek_types.ml

typer.cmo:sarek_types.cmo typer.ml

sarek_grammar.cmo:sarek_types.cmo typer.cmo gen_caml.cmo gen_kir.cmo mparser.cmo sarek_grammar.ml


test: kernels_int.cma test.ml
	@echo "(* ORIGINAL *)"
	@echo "(************)"
	@echo "(*"
	@cat test.ml
	@echo "*)"
	@echo
	@echo "(******************************************************************************)"
	@echo
	@echo "(* MODIFIED *)"
	@echo "(************)"
	@echo
	@camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma test.ml

%.pp.ml : %.ml kernels_int.cma
	camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma $< > $@


%.pp.byte: %.pp.ml
	ocamlfind ocamlc -thread -package ctypes,spoc,sarek, -linkpkg -o $@ $<

%.pp.nat: %.pp.ml
	ocamlfind ocamlopt -nodynlink -thread  -package ctypes,spoc,sarek,graphics,camlimages,tsdl,camlimages.png -linkpkg -o $@ $<

%.nat : %.ml
	ocamlfind ocamlopt -unsafe -thread -package spoc,sarek \
		-package sarek_syntax -syntax camlp4o -linkpkg -o $@ $<	

%.byte : %.ml
	ocamlfind ocamlc -thread -package spoc,sarek \
		-package sarek_syntax -syntax camlp4o -linkpkg -o $@ $<	

pp_asm: all
	camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma test.ml > test_pp.ml
	ocamlfind ocamlopt -thread -package ctypes,spoc,sarek -linkpkg test_pp.ml

clean:
	rm -rf *.cmo *.cmx *.out *~ *.cmi camlprog.exe *.cma *.asm *.pp.ml *.pp.nat *.pp.byte log
