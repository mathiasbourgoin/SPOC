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

___

## Samples :

#### [**<u>Bitonic sort</u>**][1]
#### [**<u>Image filter</u>**][2]
#### [**<u>Mandelbrot</u>**][7]
#### [**<u>Test OpenCL</u>**][8]

___

## **Tutorials :**

#### [**<u>ImageFilter</u>**](tutos/imageFilter.html)
#### [**<u>Mandelbrot</u>**](tutos/mandel.html)
#### [**<u>Sarek2CL</u>**](tutos/sarek2cl.html)


#I - How to Build SPOC



## 1 - Dependencies 

Requires :
    
    * ocaml >= 4.01.0
    * camlp4
    * ocamlfind 
    * ocp-build
    * camlp4-extra (for ubuntu)
    * m4
    
For cuda compilation :

    * nvcc    

## 2 - Compilation & Installation

### From Opam

Simply add our repository :

    opam repository add spoc_repo https://github.com/mathiasbourgoin/opam_repo_dev.git

then

    opam update spoc_repo
    opam install spoc
    

### From sources

To compile SPOC:

    cd Spoc
    make
    make install


## 3 - Build Documentation

From the sources : 

    make doc

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


#Publications#

### Talks ###
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

- *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Experiments with Spoc."**,  In Patterns for parallel programming on GPUs, F. Magoules (Ed.), Saxe-Coburg Publications, February 2013. To appear.

### Peer Reviewed Papers ###
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"High-Performance GPGPU Programming with OCaml"**, OCaml 2013, 2013
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Retour d'experience : portage d'une application haute-performance vers un langage de haut niveau"**, ComPas/RENPAR 2013, pp. 1-8 (2013)
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Spoc: GPGPU Programming Through Stream Processing with OCAML"**, Parallel Processing Letters, vol. 22 (2), pp. 1-12 (2012)
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"Experiments with Spoc."**, Workshop OpenGPU, HIPEAC 2012., Paris, France (2012)
+ *M. Bourgoin, E. Chailloux, J.\-L. Lamotte* : **"SPOC : GPGPU programming through Stream Processing with OCaml"**, HLPGPU2012 workshop, pp. 1-8 (2012)

##Contact##

UPMC - LIP6  
Boite courrier 169  
Couloir 26-00, Etage 3, Bureau 325  
4 place Jussieu  
75252 PARIS CEDEX 05  
Tel: 01 44 27 88 16, Mathias.Bourgoin (at) lip6.fr



[1]: docs/bitonic.html
[2]: docs/imageFilter.html
[7]: docs/mandelbrot.html
[8]: docs/testOpenCL.html
[3]: http://ocsigen.org/js_of_ocaml/ 
[4]: http://ftp.mozilla.org/pub/mozilla.org/firefox/releases/28.0/
[5]: http://webcl.nokiaresearch.com/
[6]: http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/
