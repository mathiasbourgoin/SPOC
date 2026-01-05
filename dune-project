(lang dune 2.9)
(name spoc)
(cram enable)

(generate_opam_files true)

(source (github mathiasbourgoin/SPOC))
(license Cecill-B)
(authors "Mathias Bourgoin")
(maintainers "mathias.bourgoin@gmail.com")

; SDK/Framework - minimal types and plugin interface, no runtime
(package
 (name spoc)
 (synopsis "GPGPU framework for OCaml - SDK types and plugin interface")
 (description
  "SPOC provides a unified interface for GPU computing in OCaml.
This package contains the SDK: framework types, device abstraction,
IR types, and plugin registration interface.
Install sarek for the full runtime and DSL.")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 2.9))
  ctypes))

; Sarek: full runtime, DSL, PPX, stdlib, and CPU backends
(package
 (name sarek)
 (synopsis "GPU kernel DSL for OCaml with runtime and CPU backends")
 (description
  "Sarek is a DSL for writing GPU kernels in OCaml syntax.
Includes the full runtime (Device, Vector, Memory management),
PPX preprocessor, standard library (Float32, Int32, etc.),
and CPU backends (native, interpreter) for development and testing.
Install sarek-cuda, sarek-opencl, or sarek-vulkan for GPU execution.")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 2.9))
  (spoc (= :version))
  (ppxlib (>= 0.22.0))
  ctypes
  ctypes-foreign
  unix
  (alcotest :with-test)))

; CUDA backend plugin
(package
 (name sarek-cuda)
 (synopsis "CUDA backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on NVIDIA GPUs via CUDA.")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 2.9))
  (sarek (= :version))
  ctypes
  ctypes-foreign)
 (depopts
  conf-cuda))

; OpenCL backend plugin
(package
 (name sarek-opencl)
 (synopsis "OpenCL backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on GPUs and CPUs via OpenCL.")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 2.9))
  (sarek (= :version))
  ctypes
  ctypes-foreign
  str)
 (depopts
  conf-opencl))

; Vulkan backend plugin
(package
 (name sarek-vulkan)
 (synopsis "Vulkan compute backend for Sarek GPU kernels")
 (description "Enables execution of Sarek kernels on GPUs via Vulkan compute shaders.")
 (depends
  (ocaml (>= 4.14.0))
  (dune (>= 2.9))
  (sarek (= :version))
  ctypes
  ctypes-foreign)
 (depopts
  conf-vulkan))
