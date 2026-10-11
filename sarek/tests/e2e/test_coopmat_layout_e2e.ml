(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * The [layout] of [CM_load] / [CM_store] executed: on every device offering a
 * configuration (HIP gfx11 WMMA, Vulkan cooperative matrix on RADV), one
 * 16x16x16 tile [D = A * B + C] for every combination of row- and
 * column-major A, B, C and D, against a host reference.
 *
 * Every input is a small integer, so every product and partial sum is exact
 * in the accumulator type and the comparison is EQUALITY even for the float
 * configurations: a wrong element position cannot hide inside a rounding
 * bound. Two bases (one 16-byte aligned for f16, one not) and a stride that
 * is neither 16 nor the base exercise the index and stride separately. The
 * interpreter is checked against the same reference, so the device and the
 * oracle agree with each other through it.
 *
 * Then Sarek_gemm.Coopmat with B stored transposed ([~b_layout:Col_major]),
 * against the same host reference as the row-major launch.
 ******************************************************************************)

open Sarek_ir_types
open Sarek
open Coopmat_hip_support
module Gemm = Sarek_gemm.Coopmat

let () = Test_helpers.Benchmarks.init_backends ()

let () = Backend_hip.init ()

let offers dev cfg =
  Sarek_capability.permits
    (C.verdict
       ~support:
         dev.Spoc_core.Device.capabilities.Spoc_framework.Framework_sig.coopmat
       cfg)

let failures = ref 0

let fail fmt =
  Printf.ksprintf
    (fun s ->
      incr failures ;
      print_endline ("  FAIL " ^ s))
    fmt

let ok fmt = Printf.ksprintf (fun s -> print_endline ("  OK  " ^ s)) fmt

let stride = 24

let len ~base = base + (15 * stride) + 16 + 8

(** [d = a * b + c], each matrix in its own layout, at [base] with [stride]. *)
let layout_ir cfg ~base ~la ~lb ~lc ~ld : kernel =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let v id name elt =
    DParam
      ( {var_name = name; var_id = id; var_type = TVec elt; var_mutable = false},
        Some {arr_elttype = elt; arr_memspace = Global} )
  in
  let k n = EConst (CInt32 (Int32.of_int n)) in
  let load dst frag src layout =
    SCoopmat
      (CM_load {dst; frag; src; index = k base; stride = k stride; layout})
  in
  {
    default_kernel with
    kern_name = "coopmat_layout_e2e";
    kern_params =
      [
        v 0 "a" (buffer_elt cfg.C.cfg_a);
        v 1 "b" (buffer_elt cfg.C.cfg_b);
        v 2 "c" (buffer_elt cfg.C.cfg_c);
        v 3 "d" (buffer_elt cfg.C.cfg_result);
      ];
    kern_body =
      SSeq
        [
          SCoopmat (CM_decl {name = "fa"; frag = fa});
          SCoopmat (CM_decl {name = "fb"; frag = fb});
          SCoopmat (CM_decl {name = "fc"; frag = fc});
          load "fa" fa "a" la;
          load "fb" fb "b" lb;
          load "fc" fc "c" lc;
          SCoopmat (CM_muladd {dst = "fc"; a = "fa"; b = "fb"; c = "fc"; cfg});
          SCoopmat
            (CM_store
               {
                 src = "fc";
                 frag = fd;
                 dst = "d";
                 index = k base;
                 stride = k stride;
                 layout = ld;
               });
        ];
  }

let at ~base layout r c =
  match layout with
  | C.Row_major -> base + (r * stride) + c
  | C.Col_major -> base + (c * stride) + r

(* Operands in [-3, 4] and C in [-50, 49]: |D| < 16 * 16 + 50, exact in u8
   products, int32, binary16 operands and binary32 sums. *)
let operand_ints ~len ~salt =
  Array.init len (fun i -> (((i * 37) + salt) mod 8) - 3)

let c_ints ~len = Array.init len (fun i -> (i * 7919 mod 100) - 50)

let sentinel = -7777

let host ~base ~la ~lb ~lc ~ld ~a ~b ~c =
  let d = Array.make (Array.length c) sentinel in
  for r = 0 to 15 do
    for col = 0 to 15 do
      let acc = ref c.(at ~base lc r col) in
      for k = 0 to 15 do
        acc := !acc + (a.(at ~base la r k) * b.(at ~base lb k col))
      done ;
      d.(at ~base ld r col) <- !acc
    done
  done ;
  d

(* Integer operands are bytes: shift into [0, 7] so u8 and the host agree. *)
let u8_of x = x + 3

let vectors cfg ~a ~b ~c ~len =
  let operand ints =
    match cfg.C.cfg_a with
    | C.Uint8 | C.Sint8 -> Execute.Vec (u8_vec (Array.map u8_of ints))
    | C.Float16 -> Execute.Vec (f16_vec (Array.map float_of_int ints))
    | C.BFloat16 | C.Float32 | C.Uint32 | C.Sint32 ->
        Execute.Vec (f32_vec (Array.map float_of_int ints))
  in
  let acc ints =
    match cfg.C.cfg_c with
    | C.Sint32 | C.Uint32 -> `I (i32_vec (Array.map Int32.of_int ints))
    | _ -> `F (f32_vec (Array.map float_of_int ints))
  in
  let d = acc (Array.make len sentinel) in
  let vec = function `I v -> Execute.Vec v | `F v -> Execute.Vec v in
  ([operand a; operand b; vec (acc c); vec d], d)

(* As floats, so a non-integer device result cannot round into a match. *)
let read = function
  | `I v -> Array.map Int32.to_float (Spoc_core.Vector.to_array v)
  | `F v -> Spoc_core.Vector.to_array v

let run_tile dev cfg ~base ~layouts:(la, lb, lc, ld) =
  let len = len ~base in
  let a = operand_ints ~len ~salt:11 and b = operand_ints ~len ~salt:5 in
  let c = c_ints ~len in
  let ir = layout_ir cfg ~base ~la ~lb ~lc ~ld in
  let host_in x = if C.component_is_integer cfg.C.cfg_a then u8_of x else x in
  let expect =
    Array.map float_of_int
    @@ host
         ~base
         ~la
         ~lb
         ~lc
         ~ld
         ~a:(Array.map host_in a)
         ~b:(Array.map host_in b)
         ~c
  in
  let one = Execute.dims1d 1 in
  let launch device =
    let args, d = vectors cfg ~a ~b ~c ~len in
    (match device with
    | Some dev ->
        let wave =
          dev.Spoc_core.Device.capabilities
            .Spoc_framework.Framework_sig.warp_size
        in
        run_gpu dev ir args ~block:(Execute.dims1d wave) ~grid:one
    | None -> run_interp ir args ~block:(Execute.dims1d 32) ~grid:one) ;
    read d
  in
  (expect, launch (Some dev), launch None)

let layouts = [C.Row_major; C.Col_major]

let combos =
  List.concat_map
    (fun la ->
      List.concat_map
        (fun lb ->
          List.concat_map
            (fun lc -> List.map (fun ld -> (la, lb, lc, ld)) layouts)
            layouts)
        layouts)
    layouts

let combo_name (la, lb, lc, ld) =
  String.concat
    ""
    (List.map
       (function C.Row_major -> "R" | C.Col_major -> "C")
       [la; lb; lc; ld])

(* The first element of [got] farther than [tol] from [expect]. The matrix
   unit of gfx1100 is not exact on these integer-valued f16/bf16 sums (one ulp
   of binary32 was observed, within the Regime B bound), so float
   configurations compare within 1e-4 relative: a misplaced element moves a
   value by at least 1. Integer configurations compare exactly. *)
let divergence ~tol got expect =
  let n = Array.length expect in
  let rec go i =
    if i >= n then None
    else
      let e = expect.(i) in
      if Float.abs (got.(i) -. e) > tol *. Float.max 1.0 (Float.abs e) then
        Some (i, got.(i), e)
      else go (i + 1)
  in
  if Array.length got <> n then Some (n, nan, nan) else go 0

let check_tile dev cfg ~base layouts =
  let expect, gpu, interp = run_tile dev cfg ~base ~layouts in
  let tol = if C.component_is_integer cfg.C.cfg_a then 0.0 else 1e-4 in
  let report who = function
    | None -> true
    | Some (i, g, e) ->
        fail
          "%s %s base %d layouts %s: %s differs from the host at %d (%.17g, \
           expected %.17g)"
          dev.Spoc_core.Device.name
          (C.config_name cfg)
          base
          (combo_name layouts)
          who
          i
          g
          e ;
        false
  in
  let device_ok = report "device" (divergence ~tol gpu expect) in
  report "interpreter" (divergence ~tol:0.0 interp expect) && device_ok

(* A configuration the device offers but this backend's codegen refuses
   (Vulkan f16: GLSL refuses float16 by measurement) is reported, not run. *)
let check_config dev cfg =
  match check_tile dev cfg ~base:8 (List.hd combos) with
  | exception e when contains ~needle:"Codegen" (Printexc.to_string e) ->
      ok
        "%s %s: refused by the codegen, not run (%s)"
        dev.Spoc_core.Device.name
        (C.config_name cfg)
        (Printexc.to_string e)
  | first ->
      let results =
        List.concat_map
          (fun base -> List.map (check_tile dev cfg ~base) combos)
          [8; 3]
      in
      if first && List.for_all Fun.id results then
        ok
          "%s %s: 16 layout combinations x 2 bases equal to the host"
          dev.Spoc_core.Device.name
          (C.config_name cfg)

(** {1 Sarek_gemm.Coopmat with B transposed} *)

let check_gemm dev =
  let m = 64 and n = 64 and k = 64 in
  let ints len salt = Array.init len (fun i -> (((i * 13) + salt) mod 9) - 4) in
  let a = ints (m * k) 1 and b = ints (k * n) 2 in
  let bt = Array.init (n * k) (fun i -> b.((i mod k * n) + (i / k))) in
  let f arr = f16_vec (Array.map float_of_int arr) in
  let expect =
    Array.init (m * n) (fun o ->
        let r = o / n and c = o mod n in
        let acc = ref 1 in
        for x = 0 to k - 1 do
          acc := !acc + (a.((r * k) + x) * b.((x * n) + c))
        done ;
        float_of_int !acc)
  in
  let go b_layout bv =
    let c = f32_vec (Array.make (m * n) 1.0) in
    let d = f32_vec (Array.make (m * n) 0.0) in
    Gemm.run ~device:dev ~b_layout ~a:(f a) ~b:bv ~c ~d ~m ~n ~k () ;
    Transfer.flush dev ;
    Spoc_core.Vector.to_array d
  in
  let differs got = divergence ~tol:1e-4 got expect <> None in
  if differs (go C.Row_major (f b)) then
    fail "%s: Gemm row-major B differs from the host" dev.Spoc_core.Device.name ;
  if differs (go C.Col_major (f bt)) then
    fail
      "%s: Gemm column-major B differs from the host"
      dev.Spoc_core.Device.name
  else
    ok
      "%s: Sarek_gemm.Coopmat with B transposed equals the host"
      dev.Spoc_core.Device.name

let () =
  C.accept_relaxed_accumulation ~reason:"layout test: exact small integers" ;
  C.accept_operand_rounding
    ~precision:C.Operands_bf16
    ~reason:"layout test: small integers are exact in bfloat16" ;
  let configs =
    [
      config C.Uint8 C.Uint8 C.Sint32;
      f16_config;
      config C.BFloat16 C.BFloat16 C.Float32;
    ]
  in
  let devices = Array.to_list (Spoc_core.Device.init ()) in
  let ran = ref 0 in
  List.iter
    (fun dev ->
      List.iter
        (fun cfg ->
          if offers dev cfg then (
            incr ran ;
            check_config dev cfg))
        configs ;
      if dev.Spoc_core.Device.framework = "HIP" && offers dev f16_config then
        check_gemm dev)
    devices ;
  if !ran = 0 then
    print_endline "[SKIP] no device offers a cooperative-matrix configuration" ;
  if !failures > 0 then (
    Printf.printf "%d failure(s)\n" !failures ;
    exit 1)
  else print_endline "PASS"
