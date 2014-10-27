P4FLAGS =  -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma 

all:
	make -j4 kernels_int.cma


install: kernels_int.cma
	ocamlfind install sarek_syntax *.cm* META

uninstall :
	ocamlfind remove sarek_syntax

kernels_int.cma:types.cmo typer.cmo gen_caml.cmo gen_kir.cmo mparser.cmo  grammar.cmo
	ocamlfind ocamlc -a $^ -o kernels_int.cma

gen_caml.cmo : types.cmo gen_caml.ml
	 ocamlfind ocamlc $(P4FLAGS)  $^ -c

gen_kir.cmo : types.cmo gen_kir.ml
	 ocamlfind ocamlc $(P4FLAGS) $^ -c

mparser.cmo:types.cmo mparser.ml
	 ocamlfind ocamlc $(P4FLAGS) $^ -c

types.cmo:types.ml
	 ocamlfind ocamlc $(P4FLAGS)  $<

typer.cmo:types.cmo typer.ml
	 ocamlfind ocamlc $(P4FLAGS)  $^ -c 

grammar.cmo:types.cmo typer.cmo gen_caml.cmo gen_kir.cmo mparser.cmo grammar.ml
	 ocamlfind ocamlc $(P4FLAGS) $^ -c


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
	ocamlfind ocamlc -thread -package ctypes,spoc,sarek,graphics,tsdl, -linkpkg -o $@ $<

%.pp.nat: %.pp.ml
	ocamlfind ocamlopt -thread -package ctypes,spoc,sarek,graphics,tsdl -linkpkg -o $@ $<

%.nat : %.ml
	ocamlfind ocamlopt -thread -package spoc,sarek,graphics,tsdl \
		-package sarek_syntax -syntax camlp4o -linkpkg -o $@ $<	

%.byte : %.ml
	ocamlfind ocamlc -thread -package spoc,sarek,graphics \
		-package sarek_syntax -syntax camlp4o -linkpkg -o $@ $<	

pp_asm: all
	camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma test.ml > test_pp.ml
	ocamlfind ocamlopt -thread -package ctypes,spoc,sarek,graphics -linkpkg test_pp.ml

clean:
	rm -rf *.cmo *,cmx *.out *~ *.cmi camlprog.exe *.cma *.asm *.pp.ml *.pp.nat *.pp.byte
