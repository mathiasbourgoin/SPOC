R=$(notdir $(CURDIR))
KERNEL_DIR= ./kernels


KERNELSSRC = $(KERNEL_DIR)/Ml_kernel.cu
NVCC = nvcc
NVCCFLAGS = -m64 -O3 -g -ptx

MLSRC=$(wildcard *.ml)
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)

NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null; echo $$?)
ifeq ($(NVCC_PRESENT), 0)
        KERNELS = $(KERNELSSRC:.cu=.ptx)
endif

#all : $(KERNELS) $(ASM) $(BYTE)
all : classifyDigitsSpoc.pp.nat

$(ASM): $(MLSRC)
	ocamlfind -g ocamlopt -thread -package spoc -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(ASM)

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -g -thread -package spoc -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(BYTE)


%.pp.nat: %.pp.ml
	ocamlfind ocamlopt -g -thread -package ctypes,ctypes.foreign,spoc,sarek,graphics -linkpkg  -o $@ $<

%.pp.ml : %.ml
	camlp4  -I +camlp4 -I `ocamlfind query spoc_external_kernels` -I `ocamlfind query sarek_syntax` -parser o -parser op -printer o kernels_int.cma -printer kernels_ext.cma $< > $@


%ptx: %cu
	$(NVCC) $(NVCCFLAGS) $(NVCC_INCLUDES) -o $@ $<

clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm kernels/*.ptx kirc_kernel.*

