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



#only compile CUDA kernels if GCC version < GCC_MAX and nvcc is installed
NVCC_PRESENT := $(shell nvcc --version 1>/dev/null 2>/dev/null;  echo $$?)
GCC_MAX := 4.3

hasKern=$(shell ls kernels 1>/dev/null 2>/dev/null; echo $$?)

ifeq ($(NVCC_PRESENT), 0)

$(info Found nvidia compiler)

CUDA_VERSION := $(shell nvcc -V | grep release | cut -f5-5 -d ' '  | sed s/,//)
$(info Cuda version $(CUDA_VERSION) is installed on the system)

ifeq ($(CUDA_VERSION), 8.0)
 GCC_MAX := 5.3
endif

ifeq ($(CUDA_VERSION), 7.0)
 GCC_MAX := 4.9
endif

ifeq ($(CUDA_VERSION), 7.0)
 GCC_MAX := 4.7
endif



GCC_VER = $(shell echo `gcc -dumpversion | cut -f1-2 -d.`)
$(info GCC version $(GCC_VER) is installed on the system)

GCC_VER_OK := $(shell echo "$(GCC_VER) <= $(GCC_MAX)" | bc)
hasKern = $(shell ls kernels 1>/dev/null 2>/dev/null; echo $$?)


ifeq ($(hasKern),0)
ifeq ($(GCC_VER_OK), 1)
KERNELS = $(KERNELSSRC:.cu=.ptx)
else
WARNING = "\033[0;31m Warning : \n Your GCC version ($(GCC_VER)) seems incompatible with your CUDA version ($(CUDA_VERSION)), \n GCC should be < $(GCC_MAX), \n CUDA kernels were not compiled \033[0m"
endif
endif
endif



all :  $(KERNELS) $(ASM) $(BYTE)
	@echo $(WARNING)



PACKAGES+=-package spoc

CLIB := $(shell ls *.c 1>/dev/null 2>/dev/null; echo $$?)

ifeq ($(CLIB), 0)
$(ASM): $(MLSRC) $(COBJS)
	ocamlfind ocamlopt -g -thread  $(PACKAGES) -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) $(COBJS) -o $(ASM)

$(BYTE): $(MLSRC) $(COBJS)
	ocamlfind ocamlc -g -custom -thread   $(PACKAGES) -linkpkg -package spoc_external_kernels -syntax camlp4o $(MLSRC) -cclib $(COBJS) -o $(BYTE)

else

$(ASM): $(MLSRC)
	ocamlfind ocamlopt -thread  $(PACKAGES) -linkpkg \
	-package spoc_external_kernels -syntax camlp4o $(MLSRC) \
	-o $(ASM)

$(BYTE): $(MLSRC)
	ocamlfind ocamlc -thread  $(PACKAGES) -linkpkg \
	-package spoc_external_kernels -syntax camlp4o $(MLSRC) \
	-o $(BYTE)

endif

%ptx: %cu
	$(NVCC) $(NVCCCFLAGS) $(NVCC_INCLUDES) -o $@ $<

%.o : %.c
	$(CC) -c $< -I `ocamlfind ocamlc -where`

clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj \
	*.asm kernels/*.ptx *.js 

