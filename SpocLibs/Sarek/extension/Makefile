all:kernels_int.cma

install: kernels_int.cma
	ocamlfind install sarek_syntax *.cm* META

uninstall :
	ocamlfind remove sarek_syntax

kernels_int.cma:typer.cmo parser.cmo  grammar.cmo
	ocamlfind ocamlc -a typer.cmo parser.cmo grammar.cmo -o kernels_int.cma

parser.cmo:parser.ml typer.cmo 
	 ocamlfind ocamlc -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  typer.cmo $<

typer.cmo:typer.ml
	 ocamlfind ocamlc -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  $<

grammar.cmo:grammar.ml typer.cmo parser.cmo
	 ocamlfind ocamlc -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma  typer.cmo parser.cmo $<


test: kernels_int.cma test.ml
	@echo "ORIGINAL"
	@echo "--------"
	@echo
	@cat test.ml
	@echo
	@echo
	@echo "--------------------------------------------------------------------------------"
	@echo
	@echo "MODIFIED"
	@echo "--------"
	@echo
	@camlp4 -I +camlp4 -parser o -parser op -printer o kernels_int.cma test.ml


clean:
	rm -rf *.cmo *,cmx *.out *~ *.cmi camlprog.exe *.cma
