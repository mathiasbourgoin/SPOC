all :
	ocamlfind ocamlopt -unsafe -thread -package spoc,sarek,graphics -linkpkg -package sarek_syntax -syntax camlp4o -o Mandelbrot.asm Mandelbrot.ml


test:
	camlp4 -I +camlp4 -I ~/.opam/4.01.0/lib/spoc -I ~/.opam/4.01.0/lib/sarek_syntax -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma Mandelbrot.ml

pp:
	camlp4 -I +camlp4 -I ~/.opam/4.01.0/lib/spoc_external_kernels -I ~/.opam/4.01.0/lib/sarek_syntax -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma Mandelbrot.ml > Mandelbrot_pp.ml
	ocamlfind ocamlopt -thread -package spoc,sarek,graphics -linkpkg -package sarek_syntax -syntax camlp4o -o Mandelbrot_pp.asm Mandelbrot_pp.ml

clean:
	rm -f *.asm *.byte *.cm* *pp.ml *.o *.cl
