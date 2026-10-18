(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Names of a kernel that the target language cannot take as written.

    OCaml lets a kernel rebind a name ([let s = s +. 1.0 in ...], two [let t] in
    one block), call a variable [half] or [x'], and the IR carries those names
    through to the generators. In C-family source the first two declare a name
    twice (or, in [float s = s + 1.0f], read the variable being declared); the
    others are keywords, types or not identifiers. The device compiler rejects
    the source, or, when the name hides a built-in function, a later call
    resolves to the variable.

    {!legalize} is a renaming that runs once at the entry of a generator, before
    any other pass. A reference resolves to its binder by [var_id] first (a
    binder of the same name that comes later in the same target block, and is
    renamed, does not capture it), then by name for what carries no identity
    (arrays, helpers, pattern variables):

    - a binder whose name is reserved for the target (per [reserved]) or already
      names a binder in scope becomes [<name>_<k>] (the smallest [k] that no
      name of the kernel uses), and every reference in its scope follows;
    - a binder that merely shadows nothing keeps its name, so a kernel without a
      clash and without a name of the reserved lists is returned physically
      unchanged and its generated source is byte-identical to before. The lists
      also hold names that compiled fine (a variable called [length], [step] or
      [dot], the [get_], [half_] and [quad_] families): a kernel using one gets
      a new source.

    Binders are [let]/[let mut] variables, [for] indices, pattern variables,
    parameters, locals, shared arrays and helper functions. Record fields,
    constructors, intrinsics, the names of record and variant types and
    cooperative-matrix fragment names are neither renamed nor avoided. A kernel
    containing [SNative] (opaque source text) is returned unchanged: this pass
    could not rename the identifiers inside it. *)

(** [legalize ~reserved ~rename_params k].

    [reserved name] says [name] cannot be declared in the target language.
    [rename_params] says whether kernel and helper parameters may be renamed
    (C-family targets: yes, an argument is bound by position). When [false],
    parameters keep their names and are not counted among the binders a local
    must avoid, because the target exposes them by name (GLSL push-constant
    macros, WGSL uniform fields) and has its own pass for a local that shadows
    one ([Sarek_ir_codegen.rename_shadowing_locals]). *)
val legalize :
  reserved:(string -> bool) ->
  rename_params:bool ->
  Sarek_ir_types.kernel ->
  Sarek_ir_types.kernel

(** {!legalize} for GLSL and WGSL: only names that are not identifiers
    ({!Sarek_reserved_words.invalid_identifier}) and locals that redeclare a
    local in scope are renamed; reserved words and parameter shadowing stay with
    the target's own escaper. *)
val legalize_shader : Sarek_ir_types.kernel -> Sarek_ir_types.kernel
