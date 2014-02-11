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
	cp _obuild/spoc_compose/*.cmi `ocamlfind query spoc_compose`

uninstall :
	ocamlfind remove spoc_compose

