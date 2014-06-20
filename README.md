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


check the [github page](http://mathiasbourgoin.github.io/SPOC/) :

      more infos
      how to build spoc
      web examples
      web tutorials
      slides from past presentations
      publications references
