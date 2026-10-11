(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Negative test (package B review H2): an explicit `let%shared` size must be a
   positive compile-time constant, like `create_array n Shared`. *)

let _bad_kernel =
  [%kernel
    fun (output : int32 vector) (n : int32) ->
      let%shared (tile : int32) = n in
      tile.(0) <- 1l ;
      output.(0) <- tile.(0)]
