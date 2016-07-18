DIR=$(notdir $(CURDIR))

MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)
PACKAGES=spoc,sarek,ctypes,dynlink,ctypes.foreign
SYNTAX_PACKAGES=spoc_external_kernels,sarek_syntax

CAMLC=ocamlfind ocamlc
CAMLOPT=ocamlfind ocamlopt

SYNTAX_INCLUDES:=-I ` ocamlfind query spoc_external_kernels` \
		 -I `ocamlfind query sarek_syntax`


all : $(KERNELS) $(ASM) $(BYTE)



$(ASM): $(MLSRC)
	$(CAMLOPT) -thread -package $(PACKAGES) -linkpkg \
	-package $(SYNTAX_PACKAGES) -syntax camlp4o -o $@ $<


$(BYTE) : $(MLSRC)
	$(CAMLC) -thread -package $(PACKAGES) -linkpkg \
	-package $(SYNTAX_PACKAGES) -syntax camlp4o -o $@ $<

test: $(MLSRC)
	camlp4 -I +camlp4 $(SYNTAX_INCLUDES) -parser o -parser op \
	-printer o kernels_int.cma -printer kernels_ext.cma $<

pp: $(DIR)_pp.asm

$(DIR)_pp.asm : $(DIR)_pp.ml
	$(CAMLOPT) -thread -package $(PACKAGES),graphics -linkpkg \
	-package $(SYNTAX_PACKAGES) -syntax camlp4o -o $@ $<


$(DIR)_pp.ml: $(MLSRC)
	camlp4 -I +camlp4 $(SYNTAX_INCLUDES)  -parser o -parser op \
	-printer o kernels_int.cma -printer kernels_ext.cma $<	> $@

clean:
	rm -f *.asm *.byte *.cm* *pp.ml *~ \#*\# *.o *.js



js:$(DIR).js

$(DIR).js: $(MLSRC) Printf_js.ml
	cat Printf_js.ml > $(DIR)_js.ml
	cat $(MLSRC) >> $(DIR)_js.ml
	ocamlfind ocamlc -thread -package $(PACKAGES),js_of_ocaml,lwt \
	-linkpkg -package $(SYNTAX_PACKAGES),js_of_ocaml.syntax \
	-syntax camlp4o $(DIR)_js.ml  -o $(DIR).js.byte
	js_of_ocaml \
	`ocamlfind query spoc`/spoc_lib.js \
	$(DIR).js.byte -o $(DIR).js


.PHONY:clean test all pp
