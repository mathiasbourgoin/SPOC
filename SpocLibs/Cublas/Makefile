all :
	ocp-build -init

clean :
	ocp-build  clean
	rm -rf tmp docs ocp-build.root.* *~ 

doc : uninstall install
	mkdir -p docs
	ocamlfind ocamldoc -thread  -html -package spoc -d docs  *.mli


install : uninstall
	ocp-build install
	cp _obuild/spoc_cublas/*.cmi `ocamlfind query spoc_cublas`

uninstall :
	ocamlfind remove spoc_cublas

