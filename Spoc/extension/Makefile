all:kernels_ext.cma

install: kernels_ext.cma
	@ocamlfind install spoc_external_kernels *.cm* META 2>>../log

uninstall :
	@(ocamlfind query spoc_external_kernels && ocamlfind remove spoc_external_kernels 2>>../log)

kernels_ext.cma:kernels_ext.ml
	@ocamlfind ocamlc -I +camlp4 -pp camlp4of dynlink.cma camlp4lib.cma $< 2>>../log
	@ocamlfind ocamlc -a kernels_ext.cmo -o kernels_ext.cma 2>>../log

clean:
	rm -rf *.cmo *,cmx *.out *~ *.cmi camlprog.exe *.cma
