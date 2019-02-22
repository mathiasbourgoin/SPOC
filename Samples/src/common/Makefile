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
# only useful with older versions of NVCC
#GCC_VER_LTE59 := $(shell echo `gcc -dumpversion | cut -f1-2 -d.` \<= 5.9 | bc )
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
CCLIBS=-cclib $(COBJS)
endif

$(ASM): $(MLSRC) $(COBJS)
	ocamlfind ocamlopt -g \
	-pp "camlp4 -I `ocamlfind query spoc_external_kernels` -parser o -printer o kernels_ext.cma" \
	-thread -package ctypes.foreign,spoc,sarek,graphics \
	`pkg-config --libs-only-L nvrtc-10.0 | sed s/"-L"/"-ccopt -L"\/g` \
	-ccopt -L$(CUDA_PATH)/lib64 -cclib -lnvrtc \
	-linkpkg  $(MLSRC) $(CCLIBS) -o $(ASM)

$(BYTE): $(MLSRC) $(COBJS)
	ocamlfind ocamlc -custom -g -ccopt "-g -rdynamic" \
	-pp "camlp4 -I `ocamlfind query spoc_external_kernels` -parser o -printer o kernels_ext.cma" \
	-thread  -package ctypes.foreign,spoc,sarek,graphics \
	`pkg-config --libs-only-L nvrtc-10.0 | sed s/"-L"/"-ccopt -L"\/g` \
	-ccopt -L$(CUDA_PATH)/lib64 -cclib -lnvrtc \
	-linkpkg $(MLSRC) $(CCLIBS) -o $(BYTE)

%ptx: %cu
	$(NVCC) $(NVCCCFLAGS) $(NVCC_INCLUDES) -o $@ $<

%.o : %.c
	$(CC) -c $< -I `ocamlfind ocamlc -where`

clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj \
	*.asm kernels/*.ptx *.js
