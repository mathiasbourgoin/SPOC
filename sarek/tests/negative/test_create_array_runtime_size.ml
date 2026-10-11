(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (package B, C-13/C-14): `create_array n Local` must be refused at PPX
   time with a located error. See the matching entry in the Makefile's
   test_negative target for the asserted message. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      let arr = create_array n Local in
      arr.(0) <- 1l ;
      output.(0) <- arr.(0)]
