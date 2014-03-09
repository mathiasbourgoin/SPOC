DIR=$(notdir $(CURDIR))

MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)
PACKAGES=spoc,sarek
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
	rm -f *.asm *.byte *.cm* *pp.ml *~ \#*\# *.o


.PHONY:clean test all pp
