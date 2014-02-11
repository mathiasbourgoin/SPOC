#I - How to Build SPOC



## 1 - Dependencies 

Requires :
    
    * ocaml >= 4.01.0
    * camlp4
    * ocamlfind 
    * ocp-build
    
For cuda compilation :

    * nvcc    

## 2 - Compilation & Installation


To compile SPOC:
cd Spoc
make
make install


## 3 - Build Documentation


make htdoc
Will build the ocamldoc html pages in the Spoc/docs directory


# II - Testing SPOC


The "Samples" directory contains few programs using SPOC.

To compile those programs:
cd Samples
make

Binaries will be located in the Samples/build folder


# III - SPOCLIBS


The "SpocLibs" directory contains few libraries based on Spoc.
 - Compose allows basic composition over GPGPU kernels
 - Cublas allows to use some functions of the Cublas library 
   (Cublas needs Cuda SDK to compile)
 - Sarek is a embedded DSL for OCaml to express kernels from the OCaml program

The Sample directory contains few samples using those libraries
