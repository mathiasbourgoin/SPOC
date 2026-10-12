(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Metal codegen for scalar f16 and for SCoopmat (simdgroup_matrix 8x8x8),
 * COMPILE-ONLY: the pure generator runs, no device is created. The executed
 * counterparts are sarek/tests/e2e/test_coopmat_metal_e2e.ml and the f16
 * tripwire, on an Apple M4 (specs/coopmat-metal-simdgroup.md).
 *
 *   1. scalar f16 is lowered: a [half] buffer, a [(half)] narrowing;
 *   2. an 8x8x8 f16*f16+f32->f32 tile is lowered to simdgroup_half8x8 /
 *      simdgroup_float8x8, with the load/store layout as the transpose flag;
 *   3. every other configuration is refused with its reason: integer (no
 *      integer simdgroup_matrix), f16 accumulation (no closed form), other
 *      shapes (MSL offers only 8x8), f32/bf16 operands, saturating, scope.
 ******************************************************************************)

open Sarek_ir_types
open Sarek_codegen
module C = Sarek_coopmat_types

let contains ~needle haystack =
  let nl = String.length needle and hl = String.length haystack in
  let rec go i =
    i + nl <= hl && (String.sub haystack i nl = needle || go (i + 1))
  in
  nl = 0 || go 0

let check_contains name ~needle out =
  if not (contains ~needle out) then
    Alcotest.failf "%s: expected %S in the output:\n%s" name needle out

let check_absent name ~needle out =
  if contains ~needle out then
    Alcotest.failf "%s: unexpected %S in the output:\n%s" name needle out

let gen k = Sarek_ir_metal.generate k

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let param name e =
  DParam (var name (TVec e), Some {arr_elttype = e; arr_memspace = Global})

(* ---- 1. scalar f16 ---------------------------------------------------- *)

let f16_scale () =
  let idx = var "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("out", EVar idx),
            ECast
              ( TFloat16,
                EBinop
                  ( Mul,
                    ECast (TFloat32, EArrayRead ("inp", EVar idx)),
                    EConst (CFloat32 1.1) ) ) ) )
  in
  {
    default_kernel with
    kern_name = "f16_scale";
    kern_params = [param "out" TFloat16; param "inp" TFloat16];
    kern_body = body;
  }

let test_f16_lowered () =
  let src = gen (f16_scale ()) in
  check_contains "f16 buffer" ~needle:"device half*" src ;
  check_contains "narrowing" ~needle:"(half)" src ;
  check_contains "widening" ~needle:"(float)" src ;
  check_absent "no coopmat header" ~needle:"simdgroup" src

(* ---- 2. coopmat tile -------------------------------------------------- *)

let shape8 = {C.m = 8; n = 8; k = 8}

let cfg ?(shape = shape8) ?(saturating = false) ?(scope = C.Subgroup) a b c r =
  {
    C.cfg_shape = shape;
    cfg_a = a;
    cfg_b = b;
    cfg_c = c;
    cfg_result = r;
    cfg_saturating = saturating;
    cfg_scope = scope;
  }

let f16_cfg = cfg C.Float16 C.Float16 C.Float32 C.Float32

let elt = function
  | C.Float16 -> TFloat16
  | C.BFloat16 | C.Float32 -> TFloat32
  | C.Uint8 | C.Sint8 -> TUint8
  | C.Uint32 | C.Sint32 -> TInt32

let tile ?(la = C.Row_major) ?(lb = C.Row_major) ?(lc = C.Row_major)
    ?(ld = C.Row_major) ?(index = 0) ?(stride = 8) cfg =
  let fa, fb, fc, fd = C.fragments_of_config cfg in
  let i = EConst (CInt32 (Int32.of_int index))
  and s = EConst (CInt32 (Int32.of_int stride)) in
  let load dst frag src layout =
    SCoopmat (CM_load {dst; frag; src; index = i; stride = s; layout})
  in
  {
    default_kernel with
    kern_name = "cm_tile";
    kern_params =
      [
        param "a" (elt cfg.C.cfg_a);
        param "b" (elt cfg.C.cfg_b);
        param "c" (elt cfg.C.cfg_c);
        param "d" (elt cfg.C.cfg_result);
      ];
    kern_body =
      SSeq
        [
          SCoopmat (CM_decl {name = "fa"; frag = fa});
          SCoopmat (CM_decl {name = "fb"; frag = fb});
          SCoopmat (CM_decl {name = "fc"; frag = fc});
          SCoopmat (CM_decl {name = "fd"; frag = fd});
          load "fa" fa "a" la;
          load "fb" fb "b" lb;
          load "fc" fc "c" lc;
          SCoopmat (CM_muladd {dst = "fd"; a = "fa"; b = "fb"; c = "fc"; cfg});
          SCoopmat
            (CM_store
               {
                 src = "fd";
                 frag = fd;
                 dst = "d";
                 index = i;
                 stride = s;
                 layout = ld;
               });
        ];
  }

let test_tile_f16 () =
  let src = gen (tile f16_cfg) in
  List.iter
    (fun needle -> check_contains "f16 tile" ~needle src)
    [
      "#include <metal_simdgroup_matrix>";
      "device half* a";
      "device float* c";
      "simdgroup_half8x8 fa = make_filled_simdgroup_matrix<half, 8, 8>(0);";
      "simdgroup_float8x8 fc = make_filled_simdgroup_matrix<float, 8, 8>(0);";
      "simdgroup_load(fa, (a + (0)), (ulong)(8), ulong2(0, 0), false);";
      "simdgroup_load(fc, (c + (0)), (ulong)(8), ulong2(0, 0), false);";
      "simdgroup_multiply_accumulate(fd, fa, fb, fc);";
      "simdgroup_store(fd, (d + (0)), (ulong)(8), ulong2(0, 0), false);";
    ]

let test_no_header_without_coopmat () =
  check_absent
    "plain kernel"
    ~needle:"metal_simdgroup_matrix"
    (gen (f16_scale ()))

let test_tile_offset_stride () =
  let src = gen (tile ~index:24 ~stride:40 f16_cfg) in
  check_contains
    "index and stride"
    ~needle:"simdgroup_load(fa, (a + (24)), (ulong)(40), ulong2(0, 0), false);"
    src

let test_tile_col_major () =
  let src =
    gen
      (tile
         ~la:C.Col_major
         ~lb:C.Col_major
         ~lc:C.Col_major
         ~ld:C.Col_major
         f16_cfg)
  in
  List.iter
    (fun needle -> check_contains "col-major" ~needle src)
    [
      "simdgroup_load(fa, (a + (0)), (ulong)(8), ulong2(0, 0), true);";
      "simdgroup_load(fb, (b + (0)), (ulong)(8), ulong2(0, 0), true);";
      "simdgroup_load(fc, (c + (0)), (ulong)(8), ulong2(0, 0), true);";
      "simdgroup_store(fd, (d + (0)), (ulong)(8), ulong2(0, 0), true);";
    ]

(* ---- 3. refusals ------------------------------------------------------ *)

let expect_refused name ~reason k =
  match gen k with
  | (_ : string) -> Alcotest.failf "%s: generation succeeded" name
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             backend = "Metal";
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {reason = got; _};
           }) ->
      if not (contains ~needle:reason got) then
        Alcotest.failf "%s: reason %S does not contain %S" name got reason
  | exception e ->
      Alcotest.failf "%s: wrong exception %s" name (Printexc.to_string e)

let test_refusals () =
  expect_refused
    "u8*u8+s32"
    ~reason:"no integer simdgroup_matrix"
    (tile (cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32)) ;
  expect_refused
    "f16 accumulate"
    ~reason:"no closed-form"
    (tile (cfg C.Float16 C.Float16 C.Float16 C.Float16)) ;
  expect_refused
    "16x16x16"
    ~reason:"only 8x8x8"
    (tile
       (cfg
          ~shape:{C.m = 16; n = 16; k = 16}
          C.Float16
          C.Float16
          C.Float32
          C.Float32)) ;
  expect_refused
    "saturating"
    ~reason:"saturating"
    (tile (cfg ~saturating:true C.Float16 C.Float16 C.Float32 C.Float32)) ;
  expect_refused
    "workgroup scope"
    ~reason:"subgroup"
    (tile (cfg ~scope:C.Workgroup C.Float16 C.Float16 C.Float32 C.Float32)) ;
  expect_refused
    "bf16"
    ~reason:"bf16"
    (tile (cfg C.BFloat16 C.BFloat16 C.Float32 C.Float32)) ;
  expect_refused
    "f32 operands"
    ~reason:"f32 operand"
    (tile (cfg C.Float32 C.Float32 C.Float32 C.Float32))

let test_buffer_mismatch () =
  let k = tile f16_cfg in
  let k =
    {
      k with
      kern_params =
        [
          param "a" TFloat32;
          param "b" TFloat16;
          param "c" TFloat32;
          param "d" TFloat32;
        ];
    }
  in
  expect_refused "wrong buffer" ~reason:"needs a float16 buffer" k

let test_undeclared_buffer () =
  let k = tile f16_cfg in
  let k =
    {k with kern_params = List.filter (fun _ -> true) (List.tl k.kern_params)}
  in
  expect_refused "undeclared buffer" ~reason:"not declared" k

(* A buffer declared inside the body, as the matrix rewrite declares its
   staging arrays (SLet of an EArrayCreate), counts when it is shared and does
   not when it is private. *)
let with_body_array space =
  let k = tile f16_cfg in
  let a = var "a" (TArray (TFloat16, space)) in
  {
    k with
    kern_params = List.tl k.kern_params;
    kern_body =
      SLet (a, EArrayCreate (TFloat16, EConst (CInt32 64l), space), k.kern_body);
  }

let test_body_shared_array () =
  let src = gen (with_body_array Shared) in
  check_contains "shared array" ~needle:"threadgroup half a[64]" src ;
  check_contains "load from it" ~needle:"simdgroup_load(fa, (a + (0))" src

let test_body_private_array () =
  expect_refused "private array" ~reason:"not declared" (with_body_array Local)

(* ---- 4. OpenCL says why it has no lowering -------------------------- *)

let test_opencl_refusal_reason () =
  let reason =
    "no OpenCL device available to this project advertises a matrix extension"
  in
  match
    Sarek_ir_opencl.generate (tile (cfg C.Uint8 C.Uint8 C.Sint32 C.Sint32))
  with
  | (_ : string) -> Alcotest.fail "OpenCL emitted a cooperative matrix"
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {reason = got; _};
             _;
           }) ->
      if not (contains ~needle:reason got) then
        Alcotest.failf "OpenCL reason %S does not say why" got
  | exception e -> Alcotest.failf "wrong exception %s" (Printexc.to_string e)

(* ---- 0. no opt-in: the lowerings are available as soon as they are used - *)

(* First in the suite and in a process that has called no opt-in: scalar f16
   and a cooperative-matrix tile generate. *)
let test_available_without_opt_in () =
  let f16 = gen (f16_scale ()) in
  if not (contains ~needle:"half" f16) then
    Alcotest.fail "scalar f16 did not generate MSL with a half type" ;
  let tile_src = gen (tile f16_cfg) in
  if not (contains ~needle:"simdgroup_multiply_accumulate" tile_src) then
    Alcotest.fail "the tile did not generate simdgroup_multiply_accumulate"

(* ---- 5. the launch block ---------------------------------------------- *)

let test_block_shape () =
  let k = tile f16_cfg in
  ignore
    (Sarek_ir_metal.generate_with_types ~block:(32, 1, 1) ~types:k.kern_types k) ;
  match
    Sarek_ir_metal.generate_with_types ~block:(16, 1, 1) ~types:k.kern_types k
  with
  | (_ : string) -> Alcotest.fail "a 16-thread block was accepted"
  | exception
      Sarek_backend_error.Backend_error.Backend_error
        (Sarek_backend_error.Backend_error.Codegen
           {
             error =
               Sarek_backend_error.Backend_error.Unsupported_construct
                 {reason; _};
             _;
           }) ->
      if not (contains ~needle:"32-wide" reason) then
        Alcotest.failf "block refusal does not name the 32-wide rule: %s" reason
  | exception e -> Alcotest.failf "wrong exception %s" (Printexc.to_string e)

let () =
  Alcotest.run
    "metal_coopmat_golden"
    [
      ( "available",
        [
          Alcotest.test_case
            "f16 and coopmat generate without an opt-in"
            `Quick
            test_available_without_opt_in;
        ] );
      ( "f16",
        [
          Alcotest.test_case "scalar f16 lowered" `Quick test_f16_lowered;
          Alcotest.test_case
            "no simdgroup header without coopmat"
            `Quick
            test_no_header_without_coopmat;
        ] );
      ( "coopmat",
        [
          Alcotest.test_case "f16 tile" `Quick test_tile_f16;
          Alcotest.test_case "index and stride" `Quick test_tile_offset_stride;
          Alcotest.test_case "column-major" `Quick test_tile_col_major;
          Alcotest.test_case "buffer mismatch" `Quick test_buffer_mismatch;
          Alcotest.test_case "undeclared buffer" `Quick test_undeclared_buffer;
          Alcotest.test_case
            "shared array declared in the body"
            `Quick
            test_body_shared_array;
          Alcotest.test_case
            "private array declared in the body"
            `Quick
            test_body_private_array;
          Alcotest.test_case "refusals" `Quick test_refusals;
        ] );
      ( "block",
        [Alcotest.test_case "32-wide SIMD-groups only" `Quick test_block_shape]
      );
      ( "opencl",
        [
          Alcotest.test_case
            "refusal names the missing extension"
            `Quick
            test_opencl_refusal_reason;
        ] );
    ]
