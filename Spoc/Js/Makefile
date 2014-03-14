FILES= SPOC.js Kernel.js Mem.js 

spoc_lib.js: $(FILES)
	echo "//concat $(FILES)" > spoc_lib.js
	for f in $(FILES) ; do \
		 echo "//# 1 \"$$f\"" >> spoc_lib.js; \
	   cat $$f >> spoc_lib.js; \
	done

clean:
	rm -f spoc_lib.js *~

install:spoc_lib.js
	cp spoc_lib.js `ocamlfind query spoc`
uninstall:
	rm -f `ocamlfind query spoc`/spoc_lib.js
