DIR=$(notdir $(CURDIR))

MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)

NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null;  echo $$?)


all : $(ASM) $(BYTE)

$(ASM): $(MLSRC)

	ocamlfind ocamlopt -thread -package spoc,graphics -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(ASM)

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -thread -package spoc,graphics -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(BYTE)


clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm kernels/*.ptx

