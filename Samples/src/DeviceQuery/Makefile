include ../common/Makefile

all : $(ASM) $(BYTE)

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


