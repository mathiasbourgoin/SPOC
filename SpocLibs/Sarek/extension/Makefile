ll:kernels_int.cma

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


compile: kernels_int.cma test.ml
	ocamlfind ocamlopt -thread -package spoc,sarek -linkpkg -package sarek_syntax,spoc_external_kernels -syntax camlp4o -o test.asm test.ml


compile_pp: test_pp.ml
	ocamlfind ocamlopt -thread -package spoc,sarek -linkpkg -package sarek_syntax,spoc_external_kernels -syntax camlp4o -o test_pp.asm test_pp.ml

test: kernels_int.cma test.ml
	@echo "(** ORIGINAL **)"
	@echo "(**************)"
	@echo
	@cat test.ml
	@echo
	@echo
	@echo "(******************************************************************************)"
	@echo
	@echo "(** MODIFIED **)"
	@echo "(**************)"
	@echo
	@camlp4 -I +camlp4 -I `ocamlfind query spoc` -I `ocamlfind query spoc_external_kernels` -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma test.ml

pp: kernels_int.cma test.ml
	@camlp4 -I +camlp4 -I `ocamlfind query spoc` -I `ocamlfind query spoc_external_kernels` -parser o -parser op -printer o kernels_ext.cma -printer kernels_int.cma test.ml > test_pp.ml 

clean:
	rm -rf *.cmo *.cmx *.out *~ *.cmi camlprog.exe *.cma *_pp.* *.o 

