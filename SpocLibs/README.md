#I - Description


The "SpocLibs" directory contains few libraries based on Spoc.
 - **Sarek** is a embedded DSL for OCaml to express kernels from the OCaml program (**experimental but tested and up to date with camlp4 and SPOC**)
 - **Benchmarks** contains few examples and benchmarks (inspired from other benchamrk suite such a rodinia) using SPOC and Sarek
 - **Cublas** allows to use some functions of the Cublas library 
   (Cublas needs Cuda SDK to compile) (**maybe deprecated but probably easy to update to current libraries**)
 - **Compose** allows basic composition over GPGPU kernels  (**mostly deprecated**)


The **Samples** directory contains few samples using those libraries. (**deprecated**)

#II - Build and Install


To build and install :

1 - make sure SPOC is correctly built and installed
2 - for each library : 	
  make
  make install
should build and install 


#III - Usage


See the **Benchmarks** and **Samples** directory for examples of programs using those libraries
