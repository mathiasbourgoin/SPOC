[![Build Status](https://github.com/mathiasbourgoin/SPOC/actions/workflows/build.yml/badge.svg)](https://github.com/mathiasbourgoin/SPOC/actions)

SPOC is a set of tools for GPGPU programming with OCaml.

The SPOC library enables the detection and use of GPGPU devices with
OCaml using Cuda and OpenCL. There is also a camlp4 syntax extension
to handle external Cuda or OpenCL kernels, as well as a DSL to express
GPGPU kernels from the OCaml code.

This work was part of my PhD thesis (UPMC-LIP6 laboratory, Paris,
France) and was partially funded by the [OpenGPU](http://opengpu.net/)
project. I continued this project in 2014-2015 in the
[Verimag](http://www-verimag.imag.fr) laboratory (Grenoble, France)
and then from 2015 to 2018 in the
[LIFO](http://www.univ-orleans.fr/lifo/) laboratory in Orléans,
France. I'm now working at [Nomadic Labs](https://nomadic-labs.com).

SPOC has been tested on multiple architectures and systems, mostly
64-bit Linux and 64-bit OSX systems. It should work with Windows too.

To be able to use SPOC, you'll need a computer capable of running
OCaml (obviously) but also compatible with either OpenCL or Cuda. For
Cuda you only need a current proprietary NVidia driver while for
OpenCL you need to install the correct OpenCL implementation for your
system. SPOC should compile anyway as everything is dynamically
linked, but you'll need Cuda/OpenCL eventually to run your programs.

# Docker image (*Probably deprecated*)
[![](https://images.microbadger.com/badges/version/mathiasbourgoin/spoc.svg)](https://microbadger.com/images/mathiasbourgoin/spoc) [![](https://images.microbadger.com/badges/image/mathiasbourgoin/spoc.svg)](https://microbadger.com/images/mathiasbourgoin/spoc)

(*The github page is largely deprecated. While it's being updated, github actions scripts (in
`.github/workflows`) may show how to build and run tests*)
For more information, examples and live tutorials,
please check the [github page](http://mathiasbourgoin.github.io/SPOC/):

      more infos
      how to build spoc
      web examples
      web tutorials
      slides from past presentations
      publications references
