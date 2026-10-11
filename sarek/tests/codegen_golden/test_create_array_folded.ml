(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Package B / PR B review H1: a constant-folded [create_array] size
   ([4l * 4l]) is accepted by the PPX, so EVERY backend must see the folded
   literal. PTX and WGSL refuse a non-literal extent; OpenCL used to emit
   [int arr[(4 * 4)]]. COMPILE-ONLY: only the pure source generators run. *)

[@@@warning "-33"]

open Sarek
module Std = Sarek_stdlib.Std

let local_kernel =
  [%kernel
    fun (out : int32 vector) ->
      let arr = create_array (4l * 4l) Local in
      arr.(0l) <- 7l ;
      out.(0l) <- arr.(0l)]

let shared_kernel =
  [%kernel
    fun (out : int32 vector) ->
      let tile = create_array (8l * 8l) Shared in
      tile.(0l) <- 7l ;
      out.(0l) <- tile.(0l)]

let ir_of (_, kirc) =
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let contains ~needle hay =
  let nn = String.length needle and nh = String.length hay in
  let rec go i = i + nn <= nh && (String.sub hay i nn = needle || go (i + 1)) in
  go 0

let failures = ref 0

let expect label cond =
  Printf.printf "  %-60s %s\n%!" label (if cond then "OK" else "FAIL") ;
  if not cond then incr failures

let attempt label f =
  match f () with
  | s -> Some s
  | exception e ->
      expect (label ^ " (raised: " ^ Printexc.to_string e ^ ")") false ;
      None

let check name k ~wgsl_needle ~opencl_needle =
  let ir = ir_of k in
  let types = ir.Sarek_ir_types.kern_types in
  (match
     attempt (name ^ " PTX") (fun () ->
         Sarek_codegen.Sarek_ir_ptx.generate_with_types ~types ir)
   with
  | Some _ -> expect (name ^ ": PTX generates") true
  | None -> ()) ;
  (match
     attempt (name ^ " WGSL") (fun () ->
         Sarek_codegen.Sarek_ir_wgsl.generate_with_types ~types ir)
   with
  | Some s ->
      expect
        (name ^ ": WGSL has " ^ wgsl_needle)
        (contains ~needle:wgsl_needle s)
  | None -> ()) ;
  match
    attempt (name ^ " OpenCL") (fun () ->
        Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types ir)
  with
  | Some s ->
      expect
        (name ^ ": OpenCL has " ^ opencl_needle)
        (contains ~needle:opencl_needle s)
  | None -> ()

let () =
  check
    "Local 4*4"
    local_kernel
    ~wgsl_needle:"array<i32, 16>"
    ~opencl_needle:"[16]" ;
  check
    "Shared 8*8"
    shared_kernel
    ~wgsl_needle:"array<i32, 64"
    ~opencl_needle:"[64]" ;
  if !failures > 0 then exit 1
