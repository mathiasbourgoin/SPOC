(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (launch bounds): a non-integer bound is refused at PPX time. See the matching entry in the Makefile's
   test_negative target for the asserted message. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      pragma ["sarek.max_threads abc"] (output.(0) <- n)]
