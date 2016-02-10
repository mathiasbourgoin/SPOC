MLSRC=$(shell ocamldep -sort *.ml)
#Kirc_Ast.ml Kirc_Cuda.ml Kirc_OpenCL.ml Kirc.ml
MLI=$(shellocamldep -sort *.mli)
#Kirc_Ast.mli Kirc.mli
MLOBJ_B= $(MLSRC:.ml=.cmo) 
MLOBJ_N=$(MLSRC:.ml=.cmx)


all : start | build 

start :
	@echo "\033[43m\033[30mBuilding Cublas\033[0m"

%.cmo:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc -c $< 

%.cmx:%.ml
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlopt -thread -package spoc -c $< 

%.cmi:%.mli
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlfind ocamlc -thread -package spoc -c $< 

%.o:%.c
	@echo  "\033[32m[$@]\033[0m" 
	@ocamlc -c -cclib -lcublas $<

cublas.cma: $(MLOBJ_B) Spoc_cublas.o
	@echo  "\033[31m[cublas.cma]\033[0m"
	ocamlfind ocamlmklib -linkall -linkpkg -cclib -lcublas_device -cclib -lcudart -cclib -lcuda Spoc_cublas.o -o cublas $(MLOBJ_B)


spoclevel-cublas : cublas.cmxa cublas.cma 
	@echo   "\033[36m[spoclevel-cublas]\033[0m"
	@ocamlmktop -custom -I +camlp4 -cclib -L. dynlink.cma \
	-cclib -L`ocamlfind query spoc` -I `ocamlfind query spoc` \
	camlp4o.cma str.cma bigarray.cma \
	-cclib -lcamlrun -cclib -lbigarray \
	spoc.cma -cclib -lspoc \
	-cclib -lcublas cublas.cma \
	-o spoclevel-cublas

cublas.cmxa : $(MLOBJ_N) Spoc_cublas.o
	@echo  "\033[31m[cublas.cmxa]\033[0m"
	ocamlfind ocamlmklib -linkall -linkpkg -cclib -lcublas_device -cclib -lcudart -cclib -lcuda Spoc_cublas.o -o cublas $(MLOBJ_N)

build: cublas.cmxa cublas.cma #spoclevel-cublas

ocp:
	ocp-build -init
	ocp-build

depend :
	ocamldep *.ml *.mli > .depend

include  .depend


ocp-clean :
	ocp-build clean	

clean :
	rm -rf tmp docs ocp-build.root.* ~* *.cm* *.o *.so log a.out spoclevel-cublas *~ cublas.a


install : all 
	@echo "\033[43m\033[30mInstalling Cublas\033[0m"
	@ocamlfind install spoc_cublas *.cma *.a  *.o *.cmxa *.cmi META 2>> log

ocp-install:
	ocp-build install
	cp _obuild/cublas/*.cm* `ocamlfind query cublas`


uninstall :
	ocamlfind remove spoc_cublas


# all :
# 	ocp-build -init

# clean :
# 	ocp-build  clean
# 	rm -rf tmp docs ocp-build.root.* *~ 

# doc : uninstall install
# 	mkdir -p docs
# 	ocamlfind ocamldoc -thread  -html -package spoc -d docs  *.mli


# install : uninstall
# 	ocp-build install
# 	cp _obuild/spoc_cublas/*.cmi `ocamlfind query spoc_cublas`

# uninstall :
# 	ocamlfind remove spoc_cublas

