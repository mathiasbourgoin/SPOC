all : asm

asm :
	ocamlfind ocamlopt  -g \
	-pp "camlp4 -I `ocamlfind query sarek_syntax` -parser o -printer o kernels_int.cma" \
	-thread -package ctypes.foreign,spoc,sarek,graphics -linkpkg -cclib -lnvrtc \
	NN.ml  -o NN.asm
#	ocamlfind ocamlopt -unsafe -thread -package spoc,sarek,graphics -linkpkg -package sarek_syntax -syntax camlp4o -o Mandelbrot.asm Mandelbrot.ml


test:
	camlp4  -I +camlp4 -I `ocamlfind query spoc_external_kernels` -I `ocamlfind query sarek_syntax` -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma NN.ml

%.pp.ml : %.ml
	camlp4  -I +camlp4 -I `ocamlfind query spoc_external_kernels` -I `ocamlfind query sarek_syntax` -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma $< > $@


%.pp.nat: %.pp.ml
	ocamlfind ocamlopt -g -thread -package ctypes,ctypes.foreign,spoc,sarek,graphics -linkpkg  -o $@ $<

pp : NN.pp.nat

clean:
	rm -f *.asm *.byte *.cm* *pp.* *.o *.cl *.json *.cu *.ptx *~  \#*\#
