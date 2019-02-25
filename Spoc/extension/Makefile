all:kernels_ext.cma

install: kernels_ext.cma
	@ocamlfind install spoc_external_kernels *.cm* META 

uninstall :
	@(ocamlfind remove spoc_external_kernels ) | true

kernels_ext.cma:kernels_ext.ml
	@ocamlfind ocamlc -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma $< 
	@ocamlfind ocamlc -a kernels_ext.cmo -o kernels_ext.cma 

clean:
	rm -rf *.cmo *,cmx *.out *~ *.cmi camlprog.exe *.cma
