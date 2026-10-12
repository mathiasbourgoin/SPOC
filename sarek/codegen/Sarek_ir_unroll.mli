(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Automatic unroll hints for small constant-trip [for] loops.

    {!annotate} wraps a [SFor] in [SPragma (["unroll"], _)] when:
    - both bounds are integer constants (literals, or [+ - *] of literals);
    - its trip count is between 1 and {!max_trip};
    - its trip count times the largest expansion of the hinted loops nested in
      its body is at most {!max_expansion} (so three nested loops of 8 hint the
      inner two only).

    A loop already governed by an explicit [SPragma] (any hint other than
    Sarek's own [sarek.*] directives) is left alone, so [pragma ["unroll 1"]]
    opts a loop out (a loop under such a pragma with a known trip count is
    counted as fully unrolled when the enclosing nest is bounded).
    [SAREK_OPT_DISABLE=unroll] in the environment
    ([Sarek_policy.restrict Auto_unroll]) makes {!annotate} the identity.
    [Execute] memoises the generated source per kernel, keyed on the policy: a
    restriction made after a kernel's first launch is honoured by its next one.

    Unrolling a constant-trip loop does not change which operations run or their
    order, so results are unchanged; only code size and register use can grow,
    which the thresholds bound. The CUDA (and HIP), OpenCL and Metal generators
    print the hint as [#pragma unroll], GLSL as [[[unroll]]]. *)

(** 16 *)
val max_trip : int

(** 64 *)
val max_expansion : int

(** Annotates the kernel body and every helper body. With [~generator] (the code
    generator's name: ["cuda-c"], ["opencl"], ["metal"], ["glsl"]) and a
    reporter installed ({!Sarek_policy.set_reporter}), reports the decision once
    per kernel and generator as an {!Sarek_policy.Auto_unroll} record whose
    device is [generator]: applied when a loop was hinted, refused when the
    restriction took away a hint. A kernel with no qualifying loop is not
    reported. *)
val annotate :
  ?generator:string -> Sarek_ir_types.kernel -> Sarek_ir_types.kernel

(** The same rule on one statement (exposed for tests). *)
val annotate_stmt : Sarek_ir_types.stmt -> Sarek_ir_types.stmt
