---
layout: index
---
[![CI](https://github.com/mathiasbourgoin/SPOC/actions/workflows/ci.yml/badge.svg)](https://github.com/mathiasbourgoin/SPOC/actions/workflows/ci.yml)


SPOC is a framework for GPGPU programming with OCaml.

Sarek is the core component: an embedded DSL for expressing GPU kernels
directly in OCaml. The SPOC framework provides the runtime infrastructure
for device detection, memory management, and kernel execution across
multiple backends (CUDA, OpenCL, Vulkan, Metal, and Native CPU).

This work was part of my PhD thesis (UPMC-LIP6 laboratory, Paris,
France) and was partially funded by the [OpenGPU](http://opengpu.net/)
project. I continued this project in 2014-2015 in the
[Verimag](http://www-verimag.imag.fr) laboratory (Grenoble, France)
and then from 2015 to 2018 in the
[LIFO](http://www.univ-orleans.fr/lifo/) laboratory in Orléans,
France. I currently work at [Nomadic Labs](https://nomadic-labs.com)

It has currently been tested on multiple architectures and systems,
mostly 64-bit Linux and 64-bit OSX systems. It should work with
Windows too.

To use SPOC, you need OCaml 5.4.0 or later. For GPU acceleration, you'll
need compatible hardware and drivers:
- **OpenCL**: Install the appropriate OpenCL runtime for your system
- **CUDA**: Requires NVIDIA GPU with current proprietary driver
- **Vulkan**: Requires Vulkan-capable GPU and drivers  
- **Metal**: macOS only, requires Metal-capable hardware
- **Native/Interpreter**: Works on any system with multicore CPU support

SPOC compiles with dynamic linking, so GPU backends are optional at build
time.

SPOC comes with some examples and I strongly advise anyone interested
to look into the slides and papers.


# Current work

Spoc and Sarek are under active development:

 - Performance optimization for Native backend
 - Enhanced custom type support (using Ctypes)
 - Improved error messages and diagnostics
 - Recursive function support in Sarek
 - CI/CD pipeline with GitHub Actions


# Docker image

CI builds use an Intel oneAPI runtime base image for OpenCL support.
The old SPOC Docker images are deprecated.


# I - How to Build SPOC



## 1 - Dependencies 

Requires:
    
    * OCaml >= 5.4.0
    * dune >= 3.0
    * ocamlfind 
    * ctypes
    * ctypes-foreign
    * ppxlib
    * alcotest (for tests)    

## 2 - Compilation & Installation

### From sources

Install dependencies:

    opam install ctypes ctypes-foreign ppxlib alcotest dune

Build and install:

    dune build
    dune install


## 3 - Build Documentation

Generate API documentation:

    dune build @doc

Documentation will be in `_build/default/_doc/_html/`


# II - Testing

Run the test suite:

    dune test

Run benchmarks:

    make benchmarks-fast

The benchmarks will automatically detect available backends and test
them.


# III - Components

The SPOC framework consists of several components:

* **Sarek** - Embedded DSL for expressing GPU kernels in OCaml
* **SPOC Framework** - Runtime infrastructure for device management and kernel execution
* **Backend plugins** - Support for CUDA, OpenCL, Vulkan, Metal, Native CPU, and Interpreter

Sarek supports custom types through Ctypes, allowing structured data in kernels.


# Publications #

### Talks ###
---
+ 2016/09/08 - [Onzième rencontre de la communauté française de compilation (Session LaMHA)](http://compilfr.ens-lyon.fr/onzieme-rencontre-compilation/) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/lahma_20160909.pdf)
+ 2015/03/10 - [Séminaire ParSys](https://www.lri.fr/seminaire.php?sem=623) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/parsys-20150310.pdf)
+ 2014/09/19 - Séminaire Synchrone - Verimag [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/verimag-20140919.pdf)
+ 2014/07/02 - [Séminaire Compilation](http://compilfr.ens-lyon.fr/huitiemes-rencontres-de-la-communaute-francaise-de-compilation/) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/compilnice2014.pdf)
+ 2014/06/13 - [Array 2014](http://www.sable.mcgill.ca/array/2014/index.html) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/array2014.pdf)
+ 2014/06/11 - [GDR GPL PhD award](http://gdr-gpl.cnrs.fr/node/132) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/gdrgpl2014.pdf)
+ 2014/04/16 - [CompSys WorkGroup](http://www.ens-lyon.fr/LIP/COMPSYS/gpu-abstract/) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/compsys-16042014.pdf)
+ 2013/12/11 - PhD defense (in french) [pdf](https://github.com/mathiasbourgoin/SPOC/blob/gh-pages/docs/talks/soutenance-11122013.pdf)
+ 2013/09/24 - [OCaml 2013](http://ocaml.org/meetings/ocaml/2013/) [pdf](http://ocaml.org/meetings/ocaml/2013/slides/bourgoin.pdf)
+ 2013/07/02 - [OCaml Users In PariS (OUPS)](http://www.meetup.com/ocaml-paris/) [pdf](http://www.algo-prog.info/spoc/docs/oups_20130702.pdf)
+ 2013/07/01 - [HLPP 2013](https://sites.google.com/site/hlpp2013/) [pdf](http://www.algo-prog.info/spoc/docs/hlpp2013.pdf)
+ 2013/05/17 - [University of Cambridge - Computer Laboratory Systems Research Group](http://talks.cam.ac.uk/talk/index/44754) [pdf](http://www.algo-prog.info/spoc/docs/cambridge-20130517.pdf)
+ 2013/01/16 - [ComPas 2013](http://compas2013.inrialpes.fr/) [pdf](http://www.algo-prog.info/spoc/docs/renpar-20130115.pdf)
+ 2012/06/21 - [GDR/GPL LAHMA 2012](http://gpl2012.irisa.fr/?q=node/30#lahma)
+ 2012/01/24 - [HiPEAC 2012 - OpenGPU Workshop](http://opengpu.net/index.php?option=com_content&view=article&id=157&Itemid=144) [pdf](http://www.algo-prog.info/spoc/docs/opengpu_20120124.pdf)
+ 2012/01/23 - [HiPEAC 2012 - HLPGPU Workshop](https://sites.google.com/site/hlpgpu/) [pdf](http://www.algo-prog.info/spoc/docs/hlpgpu_20120123.pdf)
+ 2011/11/17 - [Groupe de travail "Programmation"](http://www-apr.lip6.fr/~chaillou/Public/programmation/) [pdf](http://www.algo-prog.info/spoc/docs/gdt_20111117.pdf)
+ 2011/06/08 - [Journee Calcul Hybride: le projet OpenGPU un an plus tard](http://www.association-aristote.fr/doku.php/public/seminaires/seminaire-2011-06-08) [pdf](http://www.algo-prog.info/spoc/docs/opengpu_20110608.pdf)

### Book Chapter ###
---
- *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Experiments with Spoc."**,  In Patterns for parallel programming on GPUs, F. Magoules (Ed.), Saxe-Coburg Publications, 2015

### Peer Reviewed Papers ###
---

#### General Presentations ####
+ *M. Bourgoin, E. Chailloux, J.-L. Lamotte* : **"Efficient Abstractions for GPGPU Programming"**, International Journal of Parallel Programming, 2013.
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"High-Performance GPGPU Programming with OCaml"**, OCaml 2013, 2013
+ *M. Bourgoin, E. Chailloux, J.-L. Lamotte* : **"Efficient Abstractions for GPGPU Programming"**, International Symposium on High-level Parallel Programming and Applications (HLPP), 2013.
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"SPOC: GPGPU Programming Through Stream Processing with OCAML"**, Parallel Processing Letters, vol. 22 (2), pp. 1-12 (2012)
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"SPOC : GPGPU programming through Stream Processing with OCaml"**, HLPGPU2012 workshop, pp. 1-8 (2012)

#### On Composition and Skeletons ####
+ *M. Bourgoin, E. Chailloux* : **"GPGPU Composition with OCaml"**, Array 2014, 2014.
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Experiments with Spoc."**, Workshop OpenGPU, HIPEAC 2012., Paris, France (2012)

#### On Web Programming with SPOC and Sarek ####
+ *M. Bourgoin, E. Chailloux* : **"High-Level Accelerated Array Programming in the Web Browser"**, Array 2015, 2015.
+ *M. Bourgoin, E. Chailloux* : **"High Performance Client-Side Web Programming with SPOC and Js\_of_ocaml"**, OCaml 2014, 2014.

#### On Applications ####
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Retour d'experience : portage d'une application haute-performance vers un langage de haut niveau"**, ComPas/RENPAR 2013, pp. 1-8 (2013)


## Contact ##

Mathias Bourgoin  
mathias.bourgoin (at) gmail.com
