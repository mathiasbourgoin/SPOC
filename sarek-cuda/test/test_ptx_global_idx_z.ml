(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** PTX parity for the 3-D global position intrinsics.

    Every name in [Sarek_ir_intrinsic_dispatch.thread_intrinsic_names] must
    lower on PTX like on CUDA C, OpenCL, GLSL and Metal. [global_idx_z] used to
    be missing from the PTX table and raised
    [Ptx_codegen_error: unsupported construct: intrinsic: global_idx_z].

    Static half: every name emits PTX. Device half: a 3-D launch writes
    [(global_idx_x, global_idx_y, global_idx_z)] at a linear index derived from
    them; the result must equal both the analytic coordinates and the
    Interpreter's output. Skips the device half when no CUDA device is present
    (under ZLUDA it executes on gfx1100, not on NVIDIA hardware). *)

open Sarek_cuda
open Sarek_ir_types
module Interp = Sarek.Sarek_ir_interp

let block = (4, 2, 2)

let grid = (2, 3, 2)

let dim_x = 4 * 2

let dim_y = 2 * 3

let dim_z = 2 * 2

let total = dim_x * dim_y * dim_z

(* The Interpreter binds variables by [var_id], so ids must be distinct. *)
let next_id = ref 0

let make_var name ty =
  incr next_id ;
  {var_name = name; var_id = !next_id; var_type = ty; var_mutable = false}

let intr name = EIntrinsic ([], name, [])

let i32 n = EConst (CInt32 (Int32.of_int n))

let out_param = make_var "out" (TVec TInt32)

(* out.(3*lin + c) <- global_idx_{x,y,z} with
   lin = gx + dim_x * (gy + dim_y * gz). *)
let coords_kernel () : kernel =
  let gx = make_var "gx" TInt32 in
  let gy = make_var "gy" TInt32 in
  let gz = make_var "gz" TInt32 in
  let lin = make_var "lin" TInt32 in
  let slot c = EBinop (Add, EBinop (Mul, EVar lin, i32 3), i32 c) in
  let store c v = SAssign (LArrayElem ("out", slot c), EVar v) in
  let row_of_z = EBinop (Add, EVar gy, EBinop (Mul, i32 dim_y, EVar gz)) in
  let body =
    SLet
      ( gx,
        intr "global_idx_x",
        SLet
          ( gy,
            intr "global_idx_y",
            SLet
              ( gz,
                intr "global_idx_z",
                SLet
                  ( lin,
                    EBinop (Add, EVar gx, EBinop (Mul, i32 dim_x, row_of_z)),
                    SSeq [store 0 gx; store 1 gy; store 2 gz] ) ) ) )
  in
  {
    default_kernel with
    kern_name = "global_idx_z_probe";
    kern_params =
      [DParam (out_param, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body = body;
  }

(* One kernel per position intrinsic: out.(0) <- name. *)
let single_kernel name : kernel =
  {
    default_kernel with
    kern_name = "single_" ^ name;
    kern_params =
      [DParam (out_param, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body = SAssign (LArrayElem ("out", i32 0), intr name);
  }

let contains_sub s sub =
  let n = String.length sub in
  let found = ref false in
  for i = 0 to String.length s - n do
    if String.sub s i n = sub then found := true
  done ;
  !found

let test_every_position_intrinsic_emits () =
  List.iter
    (fun name ->
      match Sarek_codegen.Sarek_ir_ptx.generate (single_kernel name) with
      | ptx ->
          Alcotest.(check bool)
            (name ^ " emits a store")
            true
            (contains_sub ptx "st.global")
      | exception e ->
          Alcotest.failf
            "%s: PTX generation raised %s"
            name
            (Printexc.to_string e))
    Sarek_codegen.Sarek_ir_intrinsic_dispatch.thread_intrinsic_names

let test_global_idx_z_reads_z_registers () =
  let ptx =
    Sarek_codegen.Sarek_ir_ptx.generate (single_kernel "global_idx_z")
  in
  List.iter
    (fun reg ->
      Alcotest.(check bool) (reg ^ " read") true (contains_sub ptx reg))
    ["%tid.z"; "%ctaid.z"; "%ntid.z"]

let expected_coords () =
  Array.init (3 * total) (fun slot ->
      let lin = slot / 3 in
      let x = lin mod dim_x in
      let y = lin / dim_x mod dim_y in
      let z = lin / (dim_x * dim_y) in
      Int32.of_int [|x; y; z|].(slot mod 3))

let interp_coords () =
  let out = Array.make (3 * total) (Interp.VInt32 (-1l)) in
  Interp.run_kernel
    (coords_kernel ())
    ~block
    ~grid
    [("out", Interp.ArgArray out)] ;
  Array.map (function Interp.VInt32 v -> v | _ -> -2l) out

let test_interpreter_matches_analytic () =
  Alcotest.(check (array int32))
    "Interpreter == analytic coordinates"
    (expected_coords ())
    (interp_coords ())

let device_coords ptx =
  Cuda_api.Device.init () ;
  let dev = Cuda_api.Device.get 0 in
  let n = 3 * total in
  let host = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n in
  Bigarray.Array1.fill host (-1l) ;
  let d_out = Cuda_api.Memory.alloc dev n Bigarray.int32 in
  Cuda_api.Memory.host_to_device ~src:host ~dst:d_out ;
  let kernel =
    Cuda_api.Kernel.load_from_ptx dev ~name:"global_idx_z_probe" ~ptx
  in
  Cuda_api.Kernel.launch
    kernel
    ~args:
      [
        Cuda_api.Kernel.ArgBuffer d_out;
        Cuda_api.Kernel.ArgInt32 (Int32.of_int n);
      ]
    ~grid
    ~block
    ~shared_mem:0
    ~stream:None ;
  Cuda_api.Device.synchronize dev ;
  Cuda_api.Memory.device_to_host ~src:d_out ~dst:host ;
  Cuda_api.Memory.free d_out ;
  Array.init n (fun i -> host.{i})

let test_device_matches_interpreter () =
  let ptx = Sarek_codegen.Sarek_ir_ptx.generate (coords_kernel ()) in
  if not (Cuda_api.is_driver_available ()) then begin
    Printf.printf "  [SKIP] no CUDA device (PTX emission checked)\n%!" ;
    Alcotest.skip ()
  end
  else
    Alcotest.(check (array int32))
      "PTX device == Interpreter"
      (interp_coords ())
      (device_coords ptx)

let () =
  Alcotest.run
    "ptx_global_idx_z"
    [
      ( "ptx",
        [
          Alcotest.test_case
            "every position intrinsic lowers on PTX"
            `Quick
            test_every_position_intrinsic_emits;
          Alcotest.test_case
            "global_idx_z reads the z special registers"
            `Quick
            test_global_idx_z_reads_z_registers;
          Alcotest.test_case
            "Interpreter agrees with the analytic coordinates"
            `Quick
            test_interpreter_matches_analytic;
          Alcotest.test_case
            "3-D launch: PTX device == Interpreter"
            `Quick
            test_device_matches_interpreter;
        ] );
    ]
