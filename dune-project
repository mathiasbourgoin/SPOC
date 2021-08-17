(lang dune 2.9)
(name spoc)
; version field is optional

(generate_opam_files true)

(source (github mathiasbourgoin/SPOC))
(license Cecill-B)
(authors "Mathias Bourgoin")
(maintainers "mathias.bourgoin@gmail.com")

(package
 (name spoc)
 (synopsis "High-level GPGPU programming library for OCaml")
 (description "")
 (depends
 (ocaml (>= 4.07.0))
  re
  (dune (> 2.0))
  cppo
  camlp4
  ctypes
  ctypes-foreign
 ))

(package
 (name sarek)
 (synopsis "GPGPU kernel DSL for OCaml")
 (description "")
 (depends
 (ocaml (>= 4.07.0))
  spoc
  re
  (dune (> 2.0))
  cppo
  camlp4
  ctypes
  ctypes-foreign
 ))
