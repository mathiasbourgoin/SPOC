(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* A kernel calling an intrinsic declared in ANOTHER library
   (intrinsics/bits.ml, library dstest_intrinsics), compiled with that library
   in the pps list. Checks, in order:

   1. the intrinsic is registered under its real module path
      ["Dstest_intrinsics"; "Bits"], in both registries;
   2. the kernel lowered it to an intrinsic call, so the generated CUDA and
      OpenCL sources carry the declared device spelling and no OCaml path;
   3. the kernel computes the right values on every enumerated backend that
      can serve a registered intrinsic: Native and Interpreter always (the
      Interpreter through the host implementation the intrinsic registers);
      HIP, CUDA, OpenCL and Metal when present.

   Exits non-zero on any failure. *)

module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer
module Bits = Dstest_intrinsics.Bits

let failures = ref 0

let check what ok =
  Printf.printf "%s %s\n%!" (if ok then "PASS" else "FAIL") what ;
  if not ok then incr failures

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

let kernel =
  [%kernel
    fun (a : int32 vector) (b : float32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then b.(tid) <- Bits.float_of_bits a.(tid) +. 1.0]

let expected_path = ["Dstest_intrinsics"; "Bits"]

let check_registries () =
  (match
     Sarek_ppx_registry.find_intrinsic "Dstest_intrinsics.Bits.float_of_bits"
   with
  | Some info ->
      check
        (Printf.sprintf
           "PPX registry path is [%s]"
           (String.concat "; " info.Sarek_ppx_registry.ii_module))
        (info.Sarek_ppx_registry.ii_module = expected_path)
  | None -> check "PPX registry has Dstest_intrinsics.Bits.float_of_bits" false) ;
  check
    "runtime registry has float_of_bits under [Dstest_intrinsics; Bits]"
    (Sarek_registry.find_fun ~module_path:expected_path "float_of_bits" <> None)

let ir =
  let _, kirc = kernel in
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let check_device_source () =
  let cuda = Sarek_codegen.Sarek_ir_cuda.generate ir in
  let opencl = Sarek_codegen.Sarek_ir_opencl.generate ir in
  check "CUDA source calls __int_as_float" (contains cuda "__int_as_float(") ;
  check "OpenCL source calls as_float" (contains opencl "as_float(") ;
  check
    "no OCaml path leaked into device source"
    (not (contains cuda "Bits." || contains opencl "Bits."))

let n = 256

let input i = (float_of_int i *. 0.25) -. 3.0

let run_on (dev : Device.t) =
  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.float32 n in
  for i = 0 to n - 1 do
    Vector.set a i (Int32.bits_of_float (input i)) ;
    Vector.set b i 0.0
  done ;
  let threads = 64 in
  Sarek.Execute.run_vectors
    ~device:dev
    ~block:(Sarek.Execute.dims1d threads)
    ~grid:(Sarek.Execute.dims1d (n / threads))
    ~ir
    ~args:
      [
        Sarek.Execute.Vec a;
        Sarek.Execute.Vec b;
        Sarek.Execute.Int32 (Int32.of_int n);
      ]
    () ;
  Transfer.flush dev ;
  (* Every input is a multiple of 0.25 in [-3, 61]: exact in float32, and so is
     the +1, so the comparison is exact on every backend. *)
  let bad = ref 0 in
  for i = 0 to n - 1 do
    if Vector.get b i <> input i +. 1.0 then incr bad
  done ;
  check
    (Printf.sprintf
       "kernel on %s (%s): %d/%d correct"
       dev.Device.name
       dev.Device.framework
       (n - !bad)
       n)
    (!bad = 0)

let () =
  check_registries () ;
  check_device_source () ;
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init () ;
  let devs = Device.init () in
  let frameworks =
    Array.to_list devs |> List.map (fun d -> d.Device.framework)
  in
  check
    "Native and Interpreter devices enumerated"
    (List.mem "Native" frameworks && List.mem "Interpreter" frameworks) ;
  check
    "host implementation registered for the Interpreter"
    (Sarek_registry.find_host_impl ~module_path:expected_path "float_of_bits"
    <> None) ;
  Array.iter
    (fun (dev : Device.t) ->
      match dev.Device.framework with
      (* GLSL and WGSL never consult the intrinsic registry (their dispatch
         has no FFI-registry post-hook), so a registered C-family template
         cannot serve them; see sarek/ppx_intrinsic/README.md. *)
      | "Vulkan" | "WebGPU" ->
          Printf.printf
            "SKIP kernel on %s (%s): no registry path for shading languages\n%!"
            dev.Device.name
            dev.Device.framework
      (* The direct PTX emitter lowers intrinsics from its own table and
         never consults the registry either; pin that it refuses, naming
         the intrinsic, rather than producing a wrong value. *)
      | "CUDA/PTX" -> (
          match run_on dev with
          | () ->
              check
                (Printf.sprintf
                   "CUDA/PTX (%s) now runs registry intrinsics: update this \
                    test and the README"
                   dev.Device.name)
                false
          | exception Sarek_codegen.Sarek_ir_ptx_types.Ptx_codegen_error msg ->
              check
                (Printf.sprintf
                   "KNOWN-GAP CUDA/PTX (%s) refuses: %s"
                   dev.Device.name
                   msg)
                (contains msg "float_of_bits"))
      | _ -> (
          try run_on dev
          with e ->
            check
              (Printf.sprintf
                 "kernel on %s (%s) raised %s"
                 dev.Device.name
                 dev.Device.framework
                 (Printexc.to_string e))
              false))
    devs ;
  if !failures > 0 then begin
    Printf.printf "%d check(s) failed\n" !failures ;
    exit 1
  end
