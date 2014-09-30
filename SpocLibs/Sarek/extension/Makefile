all:kernels_int.cma

install: kernels_int.cma
	ocamlfind install sarek_syntax *.cm* META

uninstall :
	ocamlfind remove sarek_syntax

kernels_int.cma:typer.cmo mparser.cmo  grammar.cmo
	ocamlfind ocamlc -a typer.cmo mparser.cmo grammar.cmo -o kernels_int.cma

mparser.cmo:mparser.ml typer.cmo 
	 ocamlfind ocamlc -warn-error +a -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  typer.cmo $<

typer.cmo:typer.ml
	 ocamlfind ocamlc -warn-error +a -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  $<

grammar.cmo:grammar.ml typer.cmo mparser.cmo
	 ocamlfind ocamlc -warn-error +a -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  typer.cmo mparser.cmo $<


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

pp: 
	camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma test.ml > test_pp.ml
	ocamlfind ocamlc -thread -package ctypes,spoc -linkpkg test_pp.ml

clean:
	rm -rf *.cmo *,cmx *.out *~ *.cmi camlprog.exe *.cma *.asm *_pp.ml
