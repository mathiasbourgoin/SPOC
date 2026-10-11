(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** What matrix units could do for a recognised GEMM nest.

    Static knowledge only: which {!Sarek_coopmat_types.config} a lowering would
    use for the nest's element types, which backends can run it, and whether the
    result would stay bit-identical ({!Sarek_coopmat_types.Strict}) or only
    bounded ({!Sarek_coopmat_types.Relaxed_bounded}). Nothing here probes a
    device or changes code. Table: specs/matrix-pattern-advice.md. *)

(** The element types of A, B, the accumulator and C. *)
type element_case =
  | F16_inputs  (** float16 x float16, float32 accumulator *)
  | F32_inputs  (** float32 throughout *)
  | Int32_inputs  (** int32 throughout *)
  | No_case  (** anything else: no configuration *)

type lowering =
  | Direct  (** the operands are the configuration's operand types *)
  | Operand_rounding  (** float32 operands rounded to float16 *)
  | Byte_planes of int
      (** each int32 operand split into its four unsigned bytes; the int payload
          is the number of 8-bit products per tile *)

(** How a lowering would change results. Not
    {!Sarek_coopmat_types.accuracy_regime} on purpose: [Relaxed_operands] is a
    larger deviation than the [Relaxed_bounded] that
    {!Sarek_coopmat.accept_relaxed_accumulation} admits, and must not be
    mistaken for it. *)
type accuracy_class =
  | Exact  (** bit-identical (integer, modulo 2{^ 32}) *)
  | Relaxed_accumulation
      (** only the addition order changes: the gamma_K bound and the existing
          opt-in apply *)
  | Relaxed_operands
      (** operands are rounded to a narrower type: no existing opt-in covers it
      *)

type advice = {
  case : element_case;
  config : Sarek_coopmat_types.config option;
  lowering : lowering option;
  accuracy_class : accuracy_class option;
  accuracy : string;
  backends : (string * string) list;  (** backend, status *)
}

val element_case : Sarek_matrix_pattern.gemm -> element_case

val advise : Sarek_matrix_pattern.gemm -> advice

(** A configuration exists for this nest. *)
val has_matrix_unit : advice -> bool

(** The alert text, several lines. *)
val render :
  ?kernel_name:string -> Sarek_matrix_pattern.gemm -> advice -> string
