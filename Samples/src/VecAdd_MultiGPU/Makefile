DIR=$(notdir $(CURDIR))
KERNEL_DIR = ./kernels

KERNELSSRC = $(KERNEL_DIR)/Spoc_kernels.cu
NVCC = nvcc 
NVCCCFLAGS = -m64 -O3 -g -ptx 


MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)

NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null;  echo $$?)
ifeq ($(NVCC_PRESENT), 0)
        KERNELS = $(KERNELSSRC:.cu=.ptx)
endif


all : $(KERNELS) $(ASM) $(BYTE)

$(ASM): $(MLSRC)

	ocamlfind ocamlopt -thread -package spoc -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(ASM)

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -thread -package spoc -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -o $(BYTE)


%ptx: %cu
	$(NVCC) $(NVCCCFLAGS) $(NVCC_INCLUDES) -o $@ $<



clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm kernels/*.ptx

