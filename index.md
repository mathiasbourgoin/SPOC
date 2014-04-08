---
layout: index
---
[![Build Status](https://travis-ci.org/mathiasbourgoin/SPOC.png?branch=master)](https://travis-ci.org/mathiasbourgoin/SPOC)


SPOC is a set of tools for GPGPU programming with OCaml.

The SPOC library enables the detection and use of GPGPU devices with OCaml using Cuda and OpenCL. 
There is also a camlp4 syntax extension to handle external Cuda or OpenCL kernels, as well as a DSL to express GPGPU kernels from the OCaml code.

This work was part of my PhD thesis and was partially funded by the opengpu project. 
I'm currently in the UPMC-LIP6 laboratory.

It has currently been tested on multiple architectures and systems, mostly 64-bit Linux and 64-bit OSX systems. It should work with Windows too.

To be able to use SPOC, you'll need a computer capable of running OCaml (obviously) but also compatible with either OpenCL or Cuda. 
For Cuda you only need a current proprietary NVidia driver while for OpenCL you need to install the correct OpenCL implementation for your system. 
SPOC should compile anyway as everything is dynamically linked, but you'll need Cuda/OpenCL eventually to run your programs.

SPOC currently lacks a real tutorial, it comes with some examples and I strongly advise anyone interested to look into the slides and papers.

# Demos in your browser (experimental)

## Using WebCL and [js\_of\_ocaml][3] :

This has been tested with Firefox 26-28 and [this plugin][5]
under Windows (32bit or 64bit), Linux (Ubuntu 13.10) (32bit and 64 bit) and Mac OS/X Mavericks.

Of course, you'll also need to have OpenCL on your system.

### How to test?

You should install/have :

* [Firefox 28][4]
on your system,
* [Nokia's plugin][5]
* an OpenCL implementation for your sytem ([AMD's one][6]
should work for most multicore x86 CPUs)

### Samples :

## [**Bitonic sort**][1]
## [**Image filter**][2]
## [**Mandelbrot**][7]


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

* **Compose** allows basic composition over GPGPU kernels
* **Cublas** allows to use some functions of the Cublas library 
   (Cublas needs Cuda SDK to compile)
* **Sarek** is an *experimental* embedded DSL for OCaml to express kernels from the OCaml program

The Sample directory contains few samples using those libraries

[1]: docs/bitonic.html
[2]: docs/imageFilter.html
[7]: docs/mandelbrot.html
[3]: http://ocsigen.org/js_of_ocaml/ 
[4]: http://ftp.mozilla.org/pub/mozilla.org/firefox/releases/28.0/
[5]: http://webcl.nokiaresearch.com/
[6]: http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/
