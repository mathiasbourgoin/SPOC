(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (launch bounds): the pragma must wrap the whole kernel body. See the matching entry in the Makefile's
   test_negative target for the asserted message. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      let i = 0l in
      pragma ["sarek.max_threads 128"] (output.(i) <- n)]
