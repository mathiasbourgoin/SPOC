(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (launch bounds): near-miss spelling is refused. See the Makefile. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      pragma ["sarek.max_threads=256"] (output.(0) <- n)]
