(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Fixtures and helpers of the policy_env scenarios (test_policy_env.ml): named
   kernels for the code generators, fake devices, and the small readers the
   scenarios share. *)

[@@@warning "-33"]

[@@@alert "-sarek_matrix"]

open Sarek
open Sarek_ir_types
module Spec = Sarek_execute.Matrix_specialise
module Lp = Sarek_execute.Launch_policy
module LE = Sarek_execute.Launch_epilogue
module Cm = Sarek_coopmat
module Adv = Sarek_matrix_advice

(* ------------------------------------------------------------------------ *)
(* Fixtures                                                                  *)
(* ------------------------------------------------------------------------ *)

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let i32 n = EConst (CInt32 (Int32.of_int n))

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

(* One rounding provably equals two: no barrier with or without a grant. *)
let t2_f16_sum = narrowing_kernel "t2_f16_sum" (store (h +: g))

(* Not provable: keeps its barrier unless single rounding is granted. *)
let q4k_like =
  let m = to_float (read "q" >>: i32 4 &: i32 63) in
  narrowing_kernel
    "q4k_like"
    (store ((h *: to_float (read "q" &: i32 15)) -: (g *: m)))

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

(* A float cooperative-matrix multiply-add: the explicit coopmat gate. *)
let coopmat_kernel =
  let cfg =
    {
      Cm.cfg_shape = {Cm.m = 16; n = 16; k = 16};
      cfg_a = Cm.Float16;
      cfg_b = Cm.Float16;
      cfg_c = Cm.Float32;
      cfg_result = Cm.Float32;
      cfg_saturating = false;
      cfg_scope = Cm.Subgroup;
    }
  in
  {
    default_kernel with
    kern_name = "coopmat_f16";
    kern_body =
      SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
  }

let k_f16 =
  [%kernel
    fun (a : float16 vector)
        (b : float16 vector)
        (c : float32 vector)
        (m : int32)
        (n : int32)
        (k : int32) ->
      let row = global_idx_y in
      let col = global_idx_x in
      if row < m && col < n then begin
        let sum = mut 0.0 in
        for i = 0 to k - 1l do
          sum :=
            sum
            +. float32_of_float16 a.((row * k) + i)
               *. float32_of_float16 b.((i * n) + col)
        done ;
        c.((row * n) + col) <- sum
      end]

let gemm_ir =
  match (snd k_f16).Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

let fake_device ~id ~framework ~is_cpu : Spoc_core.Device.t =
  {
    id;
    backend_id = 0;
    name = "fake " ^ framework;
    framework;
    capabilities =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        compute_capability = (0, 0);
        device_features = [];
        coopmat = None;
        supports_atomics = false;
        warp_size = 32;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu;
      };
  }

let fake_hip = fake_device ~id:0 ~framework:"HIP" ~is_cpu:false

let fake_native = fake_device ~id:1 ~framework:"Native" ~is_cpu:true

(* ------------------------------------------------------------------------ *)
(* Helpers                                                                   *)
(* ------------------------------------------------------------------------ *)

let cuda_c k = Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] k

let opencl k = Sarek_codegen.Sarek_ir_opencl.generate_with_types ~types:[] k

let metal k = Sarek_codegen.Sarek_ir_metal.generate_with_types ~types:[] k

let glsl k = Sarek_codegen.Sarek_ir_glsl.generate_with_types ~types:[] k

let count ~needle s =
  let n = String.length needle in
  let rec go i acc =
    if i + n > String.length s then acc
    else if String.sub s i n = needle then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let contains ~needle s = count ~needle s > 0

let barriers k = count ~needle:"__float2half(sarek_f32_barrier(" (cuda_c k)

let unroll_hints () =
  List.map
    (fun gen -> count ~needle:"#pragma unroll\n" (gen unroll_loop))
    [cuda_c; opencl; metal]
  @ [count ~needle:"[[unroll]]" (glsl unroll_loop)]

let deferred ?(streamed = false) ?(soa_params = []) ?(kernel = "env_kernel")
    device =
  Lp.deferred ~kernel ~device ~streamed ~soa_params

let epilogue backend =
  LE.decide_pair ~backend ~producer:"env_p" ~consumer:"env_c"

let decision_text = function
  | Sarek_policy.Applied Sarek_policy.Exact_by_construction -> "exact"
  | Sarek_policy.Applied (Sarek_policy.Granted {reason; _}) ->
      "granted: " ^ reason
  | Sarek_policy.Applied (Sarek_policy.Proof p) -> "proof: " ^ p
  | Sarek_policy.Refused why -> "refused: " ^ why

let disabled_text = "refused: disabled (SAREK_OPT_DISABLE)"

let check_restricted expected =
  List.iter
    (fun o ->
      Alcotest.(check bool)
        (Sarek_policy.optimisation_name o)
        (List.mem o expected)
        (Sarek_policy.restricted o))
    Sarek_policy.all_optimisations

let ok_or_fail = function
  | Ok () -> ()
  | Error r -> Alcotest.fail (Sarek_policy.refusal_message r)

let single = Sarek_policy.Deviation Sarek_policy.Single_rounding

let contraction = Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction

let automatic = Sarek_policy.Consent Sarek_policy.Automatic_float_rewrite

let lowering ?precision ~explicit case =
  Spec.lowering_for
    ~budget:(Sarek_policy.budget ~kernels:[])
    ~explicit
    case
    precision

let is_ok = function Ok _ -> true | Error _ -> false

(* ------------------------------------------------------------------------ *)
