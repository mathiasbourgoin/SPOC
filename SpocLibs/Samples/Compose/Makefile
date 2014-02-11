include ../../../config

MLSRC=IteratePower.ml


MLLDFLAGSNAT = -pp "camlp4of -I $(SPOCPATH) kernels_ext.cmo" str.cmxa bigarray.cmxa  -verbose
MLLDFLAGSBYTE = -pp "camlp4of -I $(SPOCPATH) kernels_ext.cmo" str.cma bigarray.cma  -verbose
MLCLIBS = -cclib -lbigarray 


ifeq ($(UNAME),Darwin)
	CC = clang
	CCLIBS = -cclib -lstdc++ -cclib -lpthread -cclib /usr/local/cuda/lib/libcublas.dylib
	NATEXE = $(MLSRC:.ml=.nat)
	BYTEXE = $(MLSRC:.ml=.byte)
endif

ifeq ($(UNAME),Linux)
CC = clang
CCLIBS = -cclib -lpthread -I /usr/local/cuda/lib64 \
	 -cclib -lcudart \
	 -cclib -lunix
NATEXE = $(MLSRC:.ml=.nat)
BYTEXE = $(MLSRC:.ml=.byte)
endif
ifeq ($(UNAME), Cygwin)
	CC = cl /MT
	CCLIBS =
	NATEXE = $(MLSRC:.ml=.nat.exe)
	BYTEXE = $(MLSRC:.ml=.byte.exe)
endif


KERNEL_DIR = ./kernels

all: $(NATEXE) $(BYTEXE)

%nat: %cmx
	ocamlopt  $(MLLDFLAGSNAT) -I $(SPOCPATH) $(CCLIBS) -g $(MLCLIBS) spoc.cmxa compose.cmxa unix.cmxa $<   -o $@

%byte: %cmo
	ocamlc -custom $(MLLDFLAGSBYTE) -I $(SPOCPATH) $(CCLIBS) $(MLCLIBS) -g  spoc.cma compose.cma unix.cma $< -o $@

%nat.exe: %cmx
	ocamlopt $(MLLDFLAGSNAT) -I $(SPOCPATH) $(CCLIBS) -g $(MLCLIBS) spoc.cmxa compose.cmxa $<   -o $@

%byte.exe: %cmo
	ocamlc -custom $(MLLDFLAGSBYTE)  -I $(SPOCPATH) $(CCLIBS) $(MLCLIBS) -g  spoc.cma compose.cma $< -o $@


%cmo: %ml
	ocamlc $(MLLDFLAGSBYTE)  -I $(SPOCPATH) $(CCLIBS) $(MLCLIBS) -g spoc.cma compose.cma -c $< 

%cmx: %ml
	ocamlopt $(MLLDFLAGSNAT)  -I $(SPOCPATH) $(CCLIBS) $(MLCLIBS) -g spoc.cmxa compose.cmxa -c $<  

clean:
	rm -rf *.o *.cm* *.a *.cmxa *~ *.exe *.nat *.byte *.obj *.asm
