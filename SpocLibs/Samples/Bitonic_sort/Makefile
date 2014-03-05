all :
	ocamlfind ocamlopt -thread -package spoc,sarek -linkpkg -package sarek_syntax -syntax camlp4o -o Bitonic_sort.asm Bitonic_sort.ml


test:
	camlp4 -I +camlp4 -I ~/.opam/4.01.0/lib/spoc -I ~/.opam/4.01.0/lib/sarek_syntax -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma Bitonic_sort.ml

pp:
	camlp4 -I +camlp4 -I ~/.opam/4.01.0/lib/spoc_external_kernels -I ~/.opam/4.01.0/lib/sarek_syntax -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma Bitonic_sort.ml > Bitonic_sort_pp.ml
	ocamlfind ocamlopt -thread -package spoc,sarek,graphics -linkpkg -package sarek_syntax -syntax camlp4o -o Bitonic_sort_pp.asm Bitonic_sort_pp.ml

clean:
	rm -f *.asm *.byte *.cm* *pp.ml *~ \#*\# *.o
