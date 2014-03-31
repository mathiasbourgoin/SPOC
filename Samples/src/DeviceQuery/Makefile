DIR=$(notdir $(CURDIR))

MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)

NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null;  echo $$?)


all : $(ASM) $(BYTE)

$(ASM): $(MLSRC)
	ocamlfind ocamlopt -thread -package spoc,graphics -linkpkg \
	-package spoc_external_kernels -syntax camlp4o $(MLSRC) \
	-o $(ASM)

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -thread -package spoc,graphics -linkpkg \
	-package spoc_external_kernels -syntax camlp4o $(MLSRC) \
	-o $(BYTE)


clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj \
	*.asm kernels/*.ptx *.js




js:$(DIR).js

$(DIR).js: $(MLSRC) Printf_js.ml
	cat Printf_js.ml > $(DIR)_js.ml
	cat $(MLSRC) >> $(DIR)_js.ml
	ocamlfind ocamlc -thread -package spoc,js_of_ocaml,lwt \
	-linkpkg -package spoc_external_kernels,js_of_ocaml.syntax \
	-syntax camlp4o $(DIR)_js.ml  -o $(DIR).js.byte
	js_of_ocaml -noruntime \
	`ocamlfind query js_of_ocaml`/runtime.js \
	`ocamlfind query spoc`/spoc_lib.js \
	$(DIR).js.byte -o $(DIR).js
