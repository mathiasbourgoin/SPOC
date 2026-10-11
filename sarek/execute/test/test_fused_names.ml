(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Names of fused kernels (Fusion_horizontal.content_named): a digest of the
 * fused IR, so the same fusion is named alike in every process and in any
 * build order (a device compiler's on-disk cache, keyed on name and source,
 * then serves it), and two different fused kernels never share a name.
 ******************************************************************************)

open Alcotest
open Coverage_fixtures
module FH = Sarek_execute.Fusion_horizontal
module FE = Sarek_execute.Fusion_epilogue

let named prefix k =
  match FH.content_named ~prefix k with
  | Some k -> k.kern_name
  | None -> fail "no content name"

let hfused members = named "sarek_hfused_" (FH.fuse ~name:"" members)

(* Members rebuilt from scratch: equal IR, different physical values. *)
let a () = producer ~name:"a" (fun i _ -> i)

let b () = producer ~name:"b" (fun i off -> i +! off)

let c () = consumer ()

let test_deterministic () =
  let first = hfused [a (); b ()] in
  ignore (hfused [b (); c ()]) ;
  check
    string
    "same members, rebuilt, after another fusion"
    first
    (hfused [a (); b ()]) ;
  check bool "prefixed" true (String.starts_with ~prefix:"sarek_hfused_" first)

let test_distinct () =
  let names =
    [
      hfused [a (); b ()];
      hfused [b (); a ()];
      hfused [a (); c ()];
      hfused [a (); b (); c ()];
      hfused [a (); a ()];
    ]
  in
  check
    Alcotest.int
    "five groups, five names"
    5
    (List.length (List.sort_uniq compare names))

let efused consumer_ir =
  named
    "sarek_efused_"
    (FE.fuse
       ~name:""
       ~producer:(a ())
       ~p_array:"dst"
       ~consumer:consumer_ir
       ~shared:[("t", "dst")])

let test_epilogue () =
  check string "same pair, same name" (efused (c ())) (efused (c ())) ;
  check
    bool
    "guarded and unguarded consumers differ"
    true
    (efused (c ()) <> efused (consumer ~guarded:false ()))

let () =
  run
    "fused_names"
    [
      ( "names",
        [
          test_case "deterministic" `Quick test_deterministic;
          test_case "distinct" `Quick test_distinct;
          test_case "epilogue" `Quick test_epilogue;
        ] );
    ]
