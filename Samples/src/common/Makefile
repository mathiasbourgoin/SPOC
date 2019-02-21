DIR=$(notdir $(CURDIR))
KERNEL_DIR = ./kernels

KERNELSSRC = $(KERNEL_DIR)/$(DIR).cu
NVCC = nvcc 
NVCCCFLAGS = -m64 -O3 -g -ptx 


MLSRC=$(DIR).ml
ASM=$(MLSRC:.ml=.asm)
BYTE=$(MLSRC:.ml=.byte)

CSRC=$(wildcard *.c)
COBJS=$(CSRC:.c=.o)

#only compile CUDA kernels if GCC version < 5.9 and nvcc is installed
GCC_VER_LTE59 := $(shell echo `gcc -dumpversion | cut -f1-2 -d.` \<= 5.9 | bc )
NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null;  echo $$?)

hasKern=$(shell ls kernels 1>/dev/null 2>/dev/null; echo $$?)

ifeq ($(hasKern),0)
ifeq ($(GCC_VER_LTE59), 1)
ifeq ($(NVCC_PRESENT), 0)
	KERNELS = $(KERNELSSRC:.cu=.ptx)
endif
endif
endif

all : $(KERNELS) $(ASM) $(BYTE)

PACKAGES+=-package spoc

CLIB := $(shell ls *.c 1>/dev/null 2>/dev/null; echo $$?)

ifeq ($(CLIB), 0)
$(ASM): $(MLSRC) $(COBJS)
	ocamlfind ocamlopt -g -thread  $(PACKAGES) -linkpkg -package spoc_external_kernels -syntax camlp4o  $(MLSRC) $(COBJS) -cclib -lnvrtc -o $(ASM) 

$(BYTE): $(MLSRC) $(COBJS)
	ocamlfind ocamlc -g -custom -thread $(PACKAGES) -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -cclib $(COBJS) -cclib -lnvrtc -o $(BYTE)

else

$(ASM): $(MLSRC)
	ocamlfind ocamlopt  -g \
	-pp "camlp4 -I `ocamlfind query sarek_syntax` -parser o -printer o kernels_int.cma" -thread -package ctypes.foreign,spoc,sarek,graphics -linkpkg -cclib -lnvrtc $(MLSRC) \
	 -o $(ASM) 

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -g -ccopt "-g -rdynamic" -thread   $(PACKAGES) -linkpkg \
	-package spoc_external_kernels -syntax camlp4o   $(MLSRC) \
	-cclib -lnvrtc -o $(BYTE)  

endif

%ptx: %cu
	$(NVCC) $(NVCCCFLAGS) $(NVCC_INCLUDES) -o $@ $<

%.o : %.c
	$(CC) -c $< -I `ocamlfind ocamlc -where`

clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj \
	*.asm kernels/*.ptx *.js 

