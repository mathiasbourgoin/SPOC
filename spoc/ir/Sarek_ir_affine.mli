(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Affine index forms for access-pattern analysis.

    An index expression such as [(row * k) + i + off] is put in the form
    [Σ cᵢ·aᵢ + c₀] where each {e atom} [aᵢ] is a loop variable or a
    thread-varying sub-expression ([thread_idx_x], [global_thread_id / n]) and
    each coefficient [cᵢ] is a {e uniform} polynomial: an integer combination of
    products of uniform factors (scalar parameters, [block_dim_*], array
    lengths, or an opaque uniform sub-expression such as [(k + 15) / 16]).

    Uniform means "the same for every thread of a launch and every iteration of
    every loop", which is what a stride must be for a matrix unit to load a tile
    with one base address and one leading dimension.

    The form is purely syntactic: two atoms are the same atom when their
    let-inlined expressions are structurally equal. That is sound for the use
    made of it (deciding which index varies with which loop) because every atom
    is a pure function of the thread and of the loop variables. It is not a
    decision procedure for index equality: [tid / n * n + tid mod n] is not
    recognised as [tid].

    Used by {!Sarek_matrix_pattern} (specs/matrix-pattern-advice.md). *)

open Sarek_ir_types

(** {1 Uniform polynomials} *)

type monomial = {coef : int; factors : expr list}

(** A normalised polynomial: monomials sorted by factor list, no zero
    coefficient, no repeated factor list. [[]] is zero. *)
type poly = monomial list

val poly_const : int -> poly

val poly_is_zero : poly -> bool

val poly_is_one : poly -> bool

val poly_equal : poly -> poly -> bool

val poly_mul : poly -> poly -> poly

(** [poly_div p d] is the exact quotient of [p] by the single-monomial [d], or
    [None] (also when [d] is not a single monomial). *)
val poly_div : poly -> poly -> poly option

val poly_to_expr : poly -> expr

(** The value of a constant polynomial ([Some 0] for zero), [None] otherwise. *)
val poly_constant : poly -> int option

(** {1 Affine forms} *)

(** [terms] is sorted by atom and has no zero coefficient. *)
type t = {terms : (expr * poly) list; const : poly}

(** How a variable that the analysis meets in an index is to be read. *)
type leaf =
  | Uniform_leaf  (** a scalar kernel parameter or another uniform value *)
  | Atom_leaf  (** a loop variable: may appear linearly, never non-linearly *)
  | Rejected_leaf  (** a mutable variable, or one the caller cannot classify *)

(** [classify_var] classifies a variable that is not let-bound; [definition v]
    is the defining expression of an immutable let-bound variable, which is
    inlined before analysis. *)
type env = {classify_var : var -> leaf; definition : var -> expr option}

type error =
  | Not_affine of expr
      (** a product, division, modulo, shift or bit operation involving a loop
          variable *)
  | Data_dependent of expr  (** the index reads memory *)
  | Unknown_var of var  (** a {!Rejected_leaf} variable *)

(** The canonical atom of a variable (the lowering marks a loop variable mutable
    at its binder and immutable at its uses). Use it to look up a loop
    variable's coefficient. *)
val var_atom : var -> expr

(** The let-inlined expression, as atoms are compared. *)
val inline : env -> expr -> expr

val of_expr : env -> expr -> (t, error) result

(** The coefficient of an atom ({!poly_is_zero} when absent). *)
val coefficient : t -> expr -> poly

val atoms : t -> expr list

(** The form [Σ cᵢ·aᵢ] with no constant term. *)
val of_terms : (expr * poly) list -> t

(** The form without the terms of the given atoms. *)
val without : t -> expr list -> t

(** No atom: the value is uniform. *)
val is_uniform : t -> bool

val to_expr : t -> expr

(** The offending sub-expression of an error, when there is one. *)
val error_expr : error -> expr option
