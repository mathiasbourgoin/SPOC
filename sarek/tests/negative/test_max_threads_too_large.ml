(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (launch bounds): a bound above 1024 is refused at PPX time. See the matching entry in the Makefile's
   test_negative target for the asserted message. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      pragma ["sarek.max_threads 2048"] (output.(0) <- n)]
