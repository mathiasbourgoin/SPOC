---
layout: index_sample
title: Interactive Sarek
---

# Interactive Sarek with Jupyter

You can use Sarek interactively using the [ocaml-jupyter](https://github.com/akabe/ocaml-jupyter) kernel. This is an excellent way to prototype kernels, visualize GPU data, and document your experiments.

## Setup

1. **Install Jupyter and the OCaml Kernel**:
   ```bash
   pip install jupyterlab
   opam install jupyter
   ocaml-jupyter-interpreter install
   ```

2. **Install Sarek**:
   Ensure Sarek and its PPX are installed in your current switch:
   ```bash
   opam install sarek sarek-cuda  # or your preferred backend
   ```

3. **Launch Jupyter**:
   ```bash
   jupyter lab
   ```

## Using Sarek in a Notebook

In an OCaml Jupyter cell, you can load Sarek and define kernels exactly as you would in a standard project.

```ocaml
(* Load Sarek *)
#require "sarek";;
#require "sarek.ppx";;

open Sarek;;

(* Define a kernel *)
let%kernel hello_gpu (out : float32 vector) =
  let tid = get_global_id 0 in
  out.(tid) <- 42.0
```

## Visualizing Results

One of the big advantages of using Jupyter is the ability to render GPU results immediately. For example, after running a Mandelbrot kernel, you can use OCaml libraries to display the resulting `Vector` as an image directly in the notebook.

## Try it Online (Binder)

*Coming Soon: We are working on a pre-configured Binder environment so you can test Sarek kernels in your browser with zero installation.*
