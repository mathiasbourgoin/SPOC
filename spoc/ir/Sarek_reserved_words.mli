(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Identifiers a generated kernel must not declare, per source language.

    A kernel is written in OCaml, where [half], [kernel], [int], [max], [local],
    [x'] and [uint] are ordinary variable names. Emitted verbatim they are a
    keyword, a type, a qualifier, a built-in function or not an identifier at
    all in the target language, and the device compiler fails (or, for a
    built-in function, resolves a later call to the variable). The predicates
    below say which names {!Sarek_ir_names.legalize} must rename.

    Each predicate is a superset of the language's keywords: over-escaping a
    name only changes the spelling in the generated source, so the lists err on
    the side of listing. GLSL and WGSL keep their own escapers
    ([Sarek_ir_glsl.escape_glsl_name], [Sarek_ir_wgsl.escape_wgsl_name]), which
    also rewrite names at emission sites that expose a parameter through a macro
    or a struct field; for those two only {!invalid_identifier} applies. *)

(** [true] when [name] is not a C identifier: empty, starting with a digit, or
    containing a character other than letters, digits and underscore (an OCaml
    [x'] is the common case). Every target rejects such a name. *)
val invalid_identifier : string -> bool

(** Keywords and library names shared by the C-family targets: ISO C11 and C++20
    keywords, <stdint.h> types, and the <math.h>/<stdlib.h> functions a
    generated expression calls by their plain name. *)
val c_family : string -> bool

(** CUDA C++ and HIP. {!c_family} plus built-in variables ([threadIdx]), vector
    and half types, atomics. Includes {!invalid_identifier}. *)
val cuda : string -> bool

(** OpenCL C. {!c_family} plus address-space qualifiers ([local], [global],
    [kernel]), scalar, vector and image types ([half], [float4]), work-item
    built-ins and the [convert_]/[as_]/[vload]/[atomic_] families. Includes
    {!invalid_identifier}. *)
val opencl : string -> bool

(** Metal Shading Language. {!c_family} plus address spaces ([device], [thread],
    [threadgroup]), the types of [namespace metal] ([half], [float4], [half2x2])
    and the [simd_]/[simdgroup_] families. Includes {!invalid_identifier}. *)
val metal : string -> bool

(** [true] for letters, digits and underscore. *)
val is_identifier_char : char -> bool
