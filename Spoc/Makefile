all :
	ocp-build -init
	cd extension && make

clean :
	ocp-build clean
	cd extension && make clean
	rm -rf tmp docs ocp-build.root.* ~*

doc : uninstall install
	mkdir -p tmp
	for i in *.mli; do \
	echo $$i; \
	echo "open Spoc" > tmp/$$i; \
	cat $$i >> tmp/$$i; \
	done
	mkdir -p docs
	ocamlfind ocamldoc -thread  -html -package spoc -d docs  tmp/*.mli
	rm -rf tmp

install : uninstall
	ocp-build install
	cp _obuild/spoc/*.cmi `ocamlfind query spoc`
	cd extension && make install

uninstall :
	ocamlfind remove spoc
	cd extension && make uninstall
