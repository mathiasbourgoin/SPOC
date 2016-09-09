---
layout: index
---
[![Build Status](https://travis-ci.org/mathiasbourgoin/SPOC.png?branch=master)](https://travis-ci.org/mathiasbourgoin/SPOC)


SPOC is a set of tools for GPGPU programming with OCaml.

The SPOC library enables the detection and use of GPGPU devices with
OCaml using Cuda and OpenCL. There is also a camlp4 syntax extension
to handle external Cuda or OpenCL kernels, as well as a DSL (called
Sarek) to express GPGPU kernels from the OCaml code.

This work was part of my PhD thesis (done in
[UPMC-LIP6](http://www.lip6.fr/?LANG=en), Paris, France) and was
partially funded by the [OPENGPU project](http://opengpu.net/EN/).
I'm currently in the [Verimag](http://www-verimag.imag.fr/?lang=en)
laboratory (Grenoble, France).

It has currently been tested on multiple architectures and systems,
mostly 64-bit Linux and 64-bit OSX systems. It should work with
Windows too.

To be able to use SPOC, you'll need a computer capable of running
OCaml (obviously) but also compatible with either OpenCL or Cuda. For
Cuda you only need a current proprietary NVidia driver while for
OpenCL you need to install the correct OpenCL implementation for your
system.  SPOC should compile anyway as everything is dynamically
linked, but you'll need Cuda/OpenCL eventually to run your programs.

SPOC comes with some examples and I strongly advise anyone interested
to look into the slides and papers. For basic tutorials, you should
look into our [live demos](#demos).


# Current work

Spoc and Sarek are still in development, here is a list of features we
plan to add (bold ones are currently in development) :

 - **Add a performance model to Sarek**
 - **Add custom types to Sarek (using Ctypes/Js\_of\_ocaml)** ->
   [example with Cards](https://github.com/mathiasbourgoin/SPOC/blob/master/SpocLibs/Sarek/extension/belote.ml)
 - **Allow recursive functions in Sarek**
 - Enable *List* handling with Spoc and Sarek
 - Add interoperability with OpenGL

# Demos in your browser (experimental)<a name="demos"></a>

## Using WebCL and [js\_of\_ocaml][3] :

This has been tested with Firefox 26-34 and [this plugin][5]
under Windows 8.1 (32bit or 64bit), multiple Linux distributions (32bit and 64 bit) and Mac OS/X Mavericks and Yosemite.

Of course, you'll also need to have OpenCL on your system.

### How to test?

You should install/have :

* [Firefox 28-34][4]
on your system,
* [Nokia's plugin][5]
* an OpenCL implementation for your sytem ([AMD's one][6]
should work for most multicore x86 CPUs)

___

<div id="contentBox" style="margin:0px auto; display:flex; width:100%">
<div id="column1" style="float:left; margin:7%; width:40%">
<pre><h2>Samples:</h2>
<h4><a href="docs/bitonic.html"><strong><u>Bitonic sort</u></strong></a></h4>
<h4><a href="docs/imageFilter.html"><strong><u>Image filter</u></strong></a></h4>
<h4><a href="docs/mandelbrot.html"><strong><u>Mandelbrot</u></strong></a></h4>
<h4><a href="docs/testOpenCL.html"><strong><u>Test OpenCL</u></strong></a></h4>
</pre></div>
<div id="column2" style="float:left; margin:7%;width:40%;">
<pre><h2>Tutorials :</h2>
<h4><a href="tutos/array2015.html"><strong><u>Array-2015 Demo</u></strong></a></h4>
<h4><a href="tutos/OCaml2014.html"><strong><u>OCaml-2014 Demo</u></strong></a></h4>
<h4><a href="tutos/imageFilter.html"><strong><u>ImageFilter</u></strong></a></h4>
<h4><a href="tutos/mandel.html"><strong><u>Mandelbrot</u></strong></a></h4>
<h4><a href="tutos/sarek2cl.html"><strong><u>Sarek2CL</u></strong></a></h4>
</pre></div>
</div>
<hr></hr>

#I - How to Build SPOC



## 1 - Dependencies 

Requires :
    
    * ocaml >= 4.01.0 (mainly tested with ocaml 4.02.1)
    * camlp4
    * ocamlfind 
    * camlp4-extra (for ubuntu)
    * m4
    
For cuda compilation :

    * nvcc    

## 2 - Compilation & Installation

### From Opam

SPOC and Sarek should be in the opam repository. 

For development releases (more up to date but maybe instable),
simply add our repository :

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

LIFO - Bâtiment IIIA
Rue Léonard de Vinci
B.P. 6759
F-45067 ORLEANS Cedex 2
France
Mathias.Bourgoin (at) univ-orleans.fr



[1]: docs/bitonic.html
[2]: docs/imageFilter.html
[7]: docs/mandelbrot.html
[8]: docs/testOpenCL.html
[3]: http://ocsigen.org/js_of_ocaml/ 
[4]: http://ftp.mozilla.org/pub/mozilla.org/firefox/releases/34.0/
[5]: http://webcl.nokiaresearch.com/
[6]: http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/
