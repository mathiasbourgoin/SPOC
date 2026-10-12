(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Behaviour-preservation golden for the optimisation policy
 * (specs/optimisation-policy.md, milestone M1).
 *
 * Prints, to stdout, in a fixed order: the source every pure generator emits
 * for a corpus of named kernels, [Sarek_coopmat.policy_verdict] for a set of
 * configurations, and every precision-grant reader, in five process states
 * (grants are one-way, so the states are sequential):
 *
 *   S0 no grant
 *   S1 [Sarek_f16_policy.accept_single_rounding (Kernel "q4k_like")]
 *   S2 [Sarek_f16_policy.accept_single_rounding All_kernels]
 *   S3a [Sarek_coopmat.accept_relaxed_accumulation] alone (verdicts only)
 *   S3 every [Sarek_coopmat.accept_*], readers printed after each operand
 *      rounding, then a second call with another reason for three of them
 *
 * The output is compared with test_policy_golden.expected by dune. The
 * expected file was produced on the tree BEFORE the policy module existed; it
 * must not change when the grants move behind that module. In S1-S3 a source
 * identical to S0's is printed as a marker, so the differences are the lines
 * that matter.
 ******************************************************************************)

open Sarek_ir_types
module Cm = Sarek_coopmat

let var ?(mut = false) name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = mut}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let i32 n = EConst (CInt32 (Int32.of_int n))

let f32c x = EConst (CFloat32 x)

let tid = var "tid" TInt32

let read a = EArrayRead (a, EVar tid)

let widen e = ECast (TFloat32, e)

let to_float e = EIntrinsic (["Sarek_stdlib"; "Gpu"], "float", [e])

let ( *: ) a b = EBinop (Mul, a, b)

let ( +: ) a b = EBinop (Add, a, b)

let ( -: ) a b = EBinop (Sub, a, b)

let ( &: ) a b = EBinop (BitAnd, a, b)

let ( >>: ) a b = EBinop (Shr, a, b)

let store e = SAssign (LArrayElem ("out", EVar tid), ECast (TFloat16, e))

let narrowing_kernel name body =
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [
        vec "out" TFloat16;
        vec "h" TFloat16;
        vec "g" TFloat16;
        vec "q" TInt32;
        vec "x" TFloat32;
      ];
    kern_body = SLet (tid, EIntrinsic ([], "global_thread_id", []), body);
  }

let h = widen (read "h")

let g = widen (read "g")

(* Provable (T2): a sum of two binary16 values keeps no barrier, with or
   without a grant. *)
let t2_f16_sum = narrowing_kernel "t2_f16_sum" (store (h +: g))

(* Not provable (Q4_K-like): keeps its barrier until single rounding is
   granted for this kernel. *)
let q4k_like =
  let m = to_float (read "q" >>: i32 4 &: i32 63) in
  narrowing_kernel
    "q4k_like"
    (store ((h *: to_float (read "q" &: i32 15)) -: (g *: m)))

(* Not provable, a different kernel name: unchanged by [Kernel "q4k_like"]. *)
let inexact_const = narrowing_kernel "inexact_const" (store (h *: f32c 1.1))

(* A constant-trip loop: Sarek_ir_unroll hints it. *)
let unroll_loop =
  let i = var "i" TInt32 in
  {
    default_kernel with
    kern_name = "unroll_loop";
    kern_params = [vec "acc" TFloat32; vec "x" TFloat32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SFor
            ( i,
              i32 0,
              i32 8,
              Upto,
              SAssign
                ( LArrayElem ("acc", EVar tid),
                  EArrayRead ("acc", EVar tid) +: read "x" ) ) );
  }

let plain_f32 =
  {
    default_kernel with
    kern_name = "plain_f32";
    kern_params = [vec "out" TFloat32; vec "x" TFloat32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SAssign (LArrayElem ("out", EVar tid), read "x" *: f32c 2.0) );
  }

let cm_config ?(side = 16) a acc =
  {
    Cm.cfg_shape = {Cm.m = side; n = side; k = side};
    cfg_a = a;
    cfg_b = a;
    cfg_c = acc;
    cfg_result = acc;
    cfg_saturating = false;
    cfg_scope = Cm.Subgroup;
  }

let cm_elt = function
  | Cm.Float16 -> TFloat16
  | Cm.BFloat16 | Cm.Float32 -> TFloat32
  | Cm.Uint8 | Cm.Sint8 -> TUint8
  | Cm.Uint32 | Cm.Sint32 -> TInt32

(* d = a * b + c over one 16x16x16 tile, row-major. *)
let cm_kernel name cfg =
  let fa, fb, fc, fd = Cm.fragments_of_config cfg in
  let zero = i32 0 and stride = i32 16 in
  let load dst frag src =
    SCoopmat
      (CM_load {dst; frag; src; index = zero; stride; layout = Cm.Row_major})
  in
  let param nm e =
    DParam (var nm (TVec e), Some {arr_elttype = e; arr_memspace = Global})
  in
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [
        param "a" (cm_elt cfg.Cm.cfg_a);
        param "b" (cm_elt cfg.Cm.cfg_b);
        param "c" (cm_elt cfg.Cm.cfg_c);
        param "d" (cm_elt cfg.Cm.cfg_result);
      ];
    kern_body =
      SSeq
        [
          SCoopmat (CM_decl {name = "fa"; frag = fa});
          SCoopmat (CM_decl {name = "fb"; frag = fb});
          SCoopmat (CM_decl {name = "fc"; frag = fc});
          load "fa" fa "a";
          load "fb" fb "b";
          load "fc" fc "c";
          SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
          SCoopmat
            (CM_store
               {
                 src = "fc";
                 frag = fd;
                 dst = "d";
                 index = zero;
                 stride;
                 layout = Cm.Row_major;
               });
        ];
  }

let corpus =
  [
    t2_f16_sum;
    q4k_like;
    inexact_const;
    unroll_loop;
    plain_f32;
    cm_kernel "cm_f16" (cm_config Cm.Float16 Cm.Float32);
    cm_kernel "cm_i32" (cm_config Cm.Uint8 Cm.Sint32);
    (* Metal lowers 8x8x8 only. *)
    cm_kernel "cm_f32_8x8" (cm_config ~side:8 Cm.Float32 Cm.Float32);
    cm_kernel "cm_f16_8x8" (cm_config ~side:8 Cm.Float16 Cm.Float32);
  ]

let cuda_c k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

(* The CUDA-C generator is the one HIP and CUDA/C share; PTX is separate. *)
let generators : (string * (kernel -> string)) list =
  [
    ("cuda-c (HIP, CUDA/C)", cuda_c);
    ( "opencl",
      fun k -> Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types:[] k );
    ( "metal",
      fun k -> Sarek_codegen.Sarek_ir_metal.generate_with_types ~types:[] k );
    ( "glsl (Vulkan)",
      fun k -> Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types:[] k );
    ( "wgsl",
      fun k -> Sarek_codegen.Sarek_ir_wgsl.generate_with_types ~types:[] k );
    ( "ptx (CUDA/PTX)",
      fun k -> Sarek_codegen.Sarek_ir_ptx.generate_with_types ~types:[] k );
  ]

let render gen k =
  match gen k with
  | src -> src
  | exception e -> "EXN " ^ Printexc.to_string e ^ "\n"

(* Sources printed in S0, to say "unchanged" later. *)
let baseline : (string * string, string) Hashtbl.t = Hashtbl.create 64

let print_sources state =
  List.iter
    (fun k ->
      List.iter
        (fun (gname, gen) ->
          let key = (k.kern_name, gname) in
          let src = render gen k in
          Printf.printf "=== %s | %s | %s ===\n" state k.kern_name gname ;
          (match Hashtbl.find_opt baseline key with
          | Some s0 when s0 = src -> print_string "(same as S0)\n"
          | _ -> print_string src) ;
          if state = "S0" then Hashtbl.replace baseline key src)
        generators)
    corpus

let verdict_text = function
  | Sarek_capability.Available -> "Available"
  | Sarek_capability.Unknown why -> "Unknown: " ^ why
  | Sarek_capability.Unavailable cap ->
      Printf.sprintf
        "Unavailable [%s/%s] %s | remedy=%s"
        cap.Sarek_capability.cap_name
        (Sarek_capability.kind_name cap.cap_kind)
        cap.cap_why
        (Option.value cap.cap_remedy ~default:"-")

let verdict_configs =
  [
    cm_config Cm.Float16 Cm.Float32;
    cm_config Cm.BFloat16 Cm.Float32;
    cm_config Cm.Uint8 Cm.Sint32;
    cm_config Cm.Sint8 Cm.Sint32;
    cm_config Cm.Float32 Cm.Float32;
    cm_config Cm.Float16 Cm.Float16;
    cm_config Cm.Float16 Cm.Sint32;
    {(cm_config Cm.Float16 Cm.Float32) with Cm.cfg_saturating = true};
  ]

let opt_text = function None -> "none" | Some r -> "Some " ^ r

let precisions = [Cm.Operands_f16; Cm.Operands_bf16; Cm.Operands_bf16x3]

let print_verdict_lines state =
  List.iter
    (fun cfg ->
      Printf.printf
        "verdict %s | %s | %s\n"
        state
        (Cm.config_name cfg)
        (verdict_text (Cm.policy_verdict cfg)))
    verdict_configs

let print_readers state =
  Printf.printf
    "reader %s | relaxed_accumulation_reason = %s\n"
    state
    (opt_text (Cm.relaxed_accumulation_reason ())) ;
  List.iter
    (fun p ->
      Printf.printf
        "reader %s | operand_rounding_reason %s = %s\n"
        state
        (Cm.operand_precision_name p)
        (opt_text (Cm.operand_rounding_reason p)))
    precisions ;
  Printf.printf
    "reader %s | automatic_float_rewrite_reason = %s\n"
    state
    (opt_text (Cm.automatic_float_rewrite_reason ())) ;
  List.iter
    (fun k ->
      Printf.printf
        "reader %s | single_rounding_reason %s = %s\n"
        state
        k.kern_name
        (opt_text (Sarek_f16_policy.single_rounding_reason ~kernel:k.kern_name)))
    corpus

let print_verdicts state =
  print_verdict_lines state ;
  print_readers state

let print_state state =
  print_sources state ;
  print_verdicts state

let result_text = function
  | Ok () -> "Ok"
  | Error r -> "Error " ^ Sarek_f16_policy.refusal_message r

let invalid_argument_text f =
  match f () with
  | () -> "no exception"
  | exception Invalid_argument m -> "Invalid_argument " ^ m

(* Refusals record nothing, so they may be exercised in any state. *)
let print_refusals () =
  let single scope reason =
    Sarek_f16_policy.accept_single_rounding scope ~reason
  in
  Printf.printf
    "refusal blank All_kernels = %s\n"
    (result_text (single Sarek_f16_policy.All_kernels " \t")) ;
  Printf.printf
    "refusal blank Kernel = %s\n"
    (result_text (single (Sarek_f16_policy.Kernel "q4k_like") "")) ;
  Printf.printf
    "refusal Kernel sarek_kern = %s\n"
    (result_text (single (Sarek_f16_policy.Kernel "sarek_kern") "a reason")) ;
  Printf.printf
    "refusal accept_relaxed_accumulation blank = %s\n"
    (invalid_argument_text (fun () ->
         Cm.accept_relaxed_accumulation ~reason:"  ")) ;
  List.iter
    (fun p ->
      Printf.printf
        "refusal accept_operand_rounding %s blank = %s\n"
        (Cm.operand_precision_name p)
        (invalid_argument_text (fun () ->
             Cm.accept_operand_rounding ~precision:p ~reason:"")))
    precisions ;
  Printf.printf
    "refusal accept_automatic_float_rewrite blank = %s\n"
    (invalid_argument_text (fun () ->
         Cm.accept_automatic_float_rewrite ~reason:"\n"))

let grant_or_fail scope reason =
  match Sarek_f16_policy.accept_single_rounding scope ~reason with
  | Ok () -> ()
  | Error r -> failwith (Sarek_f16_policy.refusal_message r)

let () =
  print_refusals () ;
  print_state "S0" ;
  grant_or_fail (Sarek_f16_policy.Kernel "q4k_like") "golden S1" ;
  print_state "S1" ;
  grant_or_fail Sarek_f16_policy.All_kernels "golden S2" ;
  print_state "S2" ;
  Cm.accept_relaxed_accumulation ~reason:"golden S3 accumulation" ;
  (* S3a: accumulation order alone; bf16 still needs operand rounding. *)
  print_verdicts "S3a" ;
  (* Readers after each single grant: independence of the precisions does not
     depend on the order they are granted in. *)
  List.iter
    (fun precision ->
      Cm.accept_operand_rounding
        ~precision
        ~reason:("golden S3 " ^ Cm.operand_precision_name precision) ;
      print_readers ("S3+" ^ Cm.operand_precision_name precision))
    precisions ;
  Cm.accept_automatic_float_rewrite ~reason:"golden S3 rewrite" ;
  (* A second call with another reason: the first reason stays. *)
  Cm.accept_relaxed_accumulation ~reason:"second accumulation reason" ;
  Cm.accept_operand_rounding ~precision:Cm.Operands_f16 ~reason:"second f16" ;
  Cm.accept_automatic_float_rewrite ~reason:"second rewrite reason" ;
  print_state "S3"
