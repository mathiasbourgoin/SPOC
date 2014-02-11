#I - Description


The "SpocLibs" directory contains few libraries based on Spoc.
 - Compose allows basic composition over GPGPU kernels
 - Cublas allows to use some functions of the Cublas library 
   (Cublas needs Cuda SDK to compile)
 - Sarek is a embedded DSL for OCaml to express kernels from the OCaml program

The Sample directory contains few smaples using those libraries


#II - Build and Install


To build and install :

1 - make sure SPOC is correctly built and installed
2 - for each library : 	
  make
  make install
should build and install 


#III - Usage


See the Sample directory for examples of programs using those libraries