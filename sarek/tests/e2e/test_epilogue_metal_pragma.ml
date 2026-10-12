(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal epilogue fusion is on by default because every Metal kernel starts
 * with [#pragma METAL fp contract(off)] (docs/fp-contraction-policy.md,
 * sections 10 and 14). This checks, on any host and without a device (compiler
 * output only), that the pragma is in the generated source of
 *   - an epilogue-fused kernel (producer + consumer), and
 *   - a horizontally fused kernel built from an epilogue-fused kernel and an
 *     independent one,
 * that the pragma precedes the first kernel (it is file scoped), that the same
 * check fails on a source without it (positive control), and that Metal refuses
 * the f16 producer (so test_epilogue_exact has no f16 pair to run there).
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module M = Sarek_codegen.Sarek_ir_metal

let producer =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (t : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then t.(i) <- a.(i) *. b.(i)]

let consumer =
  [%kernel
    fun (o : float32 vector)
        (t : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- t.(i) +. c.(i)]

let independent =
  [%kernel
    fun (x : float32 vector) (y : float32 vector) (n : int32) ->
      let i = global_thread_id in
      if i < n then y.(i) <- x.(i) *. 2.0]

let producer_f16 =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (t : float16 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then t.(i) <- float16_of_float32 (a.(i) *. b.(i))]

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let find_sub ~sub src =
  let n = String.length sub and m = String.length src in
  let rec go i =
    if i + n > m then None
    else if String.sub src i n = sub then Some i
    else go (i + 1)
  in
  go 0

let pragma = String.trim M.metal_fp_contract_pragma

(* The pragma is file scoped, so it must precede the first kernel. *)
let pragma_before_kernel src =
  match (find_sub ~sub:pragma src, find_sub ~sub:"kernel void" src) with
  | Some p, Some k -> p < k
  | _ -> false

let strip_pragma src =
  match find_sub ~sub:pragma src with
  | None -> src
  | Some p ->
      let rest = p + String.length pragma in
      String.sub src 0 p ^ String.sub src rest (String.length src - rest)

let epilogue_fused () =
  Sarek_execute.Fusion_epilogue.fuse
    ~name:"epi_pragma"
    ~producer:(ir_of producer)
    ~p_array:"t"
    ~consumer:(ir_of consumer)
    ~shared:[("t", "t")]

let checks () =
  let epi = epilogue_fused () in
  let epi_src = M.generate epi in
  let horizontal_src =
    M.generate
      (Sarek_execute.Fusion_horizontal.fuse
         ~name:"epi_then_horizontal"
         [epi; ir_of independent])
  in
  let refuses_f16 =
    match M.generate (ir_of producer_f16) with
    | _ -> false
    | exception e -> find_sub ~sub:"f16" (Printexc.to_string e) <> None
  in
  [
    ( "epilogue-fused MSL: pragma before the kernel",
      pragma_before_kernel epi_src );
    ( "epilogue then horizontal MSL: pragma before the kernel",
      pragma_before_kernel horizontal_src );
    ( "control: epilogue-fused source without the pragma fails",
      not (pragma_before_kernel (strip_pragma epi_src)) );
    ( "control: horizontal source without the pragma fails",
      not (pragma_before_kernel (strip_pragma horizontal_src)) );
    ("Metal refuses the f16 producer", refuses_f16);
  ]

let () =
  print_endline "=== Metal epilogue fusion: fp-contract pragma ===" ;
  let failed = ref [] in
  List.iter
    (fun (label, ok) ->
      Printf.printf "  %-62s %s\n%!" label (if ok then "PASS" else "FAIL") ;
      if not ok then failed := label :: !failed)
    (checks ()) ;
  if !failed <> [] then begin
    Printf.printf
      "test_epilogue_metal_pragma: FAILED (%s)\n"
      (String.concat ", " (List.rev !failed)) ;
    exit 1
  end ;
  print_endline "test_epilogue_metal_pragma: PASSED"
