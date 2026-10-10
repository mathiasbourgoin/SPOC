(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Expected error: "Unknown kernel function or constant Bits.float_of_bits"

   The kernel calls an intrinsic declared with let%sarek_intrinsic in another
   library (sarek/tests/downstream/intrinsics, dstest_intrinsics). That library
   is a link dependency here but NOT in the pps list, so the kernel PPX never
   saw its registration. This used to compile: the typer accepted any unbound
   [M.f] with a fresh type, and every source backend printed it verbatim, so
   the device compiler received [Bits.float_of_bits(...)] and failed only at
   JIT time. It must now be refused at PPX time, naming the call and the fix.
   The positive counterpart is
   sarek/tests/downstream/test_downstream_intrinsic.ml. *)

module Bits = Dstest_intrinsics.Bits

let _k =
  [%kernel
    fun (a : int32 vector) (b : float32 vector) ->
      let i = thread_idx_x in
      b.(i) <- Bits.float_of_bits a.(i)]
