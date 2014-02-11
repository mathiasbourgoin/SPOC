all :
	ocp-build -init
	cd extension && make

clean :
	ocp-build clean
	cd extension && make clean
	rm -rf _obuild ocp-build.root.* *~ 

install : uninstall
	ocp-build install
	cp _obuild/sarek/*.cm* `ocamlfind query sarek`
	cd extension && make install

uninstall :
	ocamlfind remove sarek
	cd extension && make uninstall
