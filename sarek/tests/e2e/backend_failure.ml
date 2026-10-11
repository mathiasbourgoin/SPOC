(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared classification of a failure raised while a test drives a device.

   Rule: only "the backend is not there" may skip a test. Everything else
   raised on a device path is a regression and must FAIL: codegen errors, the
   device compiler or module loader rejecting the source, launch failures,
   [Feature_not_supported] (raised mid-launch by the plugins), allocation and
   copy failures, the runtime-compile bindings' [Hiprtc_error] / [Nvrtc_error]
   (which reach the caller unwrapped, runtime review R-40), and any exception
   from the Sarek runtime itself ([Failure], [Invalid_argument], Kirc, Fusion,
   Execute, an unwrapped [Ptx_codegen_error]). Without this a CI runner whose
   first device is CPU OpenCL turns invalid generated OpenCL C into a green
   "SKIPPED (backend error)" (docs review D-40, D-41). *)

module E = Sarek_backend_error.Backend_error

(* The backend, library or device simply is not there. Nothing else skips. *)
let is_unavailable = function
  | E.Backend_error
      ( Plugin {error = Backend_unavailable _ | Library_not_found _; _}
      | Runtime {error = No_device_selected _ | Device_not_found _; _} ) ->
      true
  | _ -> false

let verdict exn = if is_unavailable exn then `Skip else `Fail

(* Stand-in for the runtime-compile binding's exception, so the injection can
   exercise the unwrapped-exception path without linking the HIP plugin. *)
module Rtc_stub = struct
  exception Hiprtc_error of string
end

let runtime backend error = E.Backend_error (E.Runtime {backend; error})

let plugin backend error = E.Backend_error (E.Plugin {backend; error})

(* The exception a given injection shape raises. *)
let injected_exn shape =
  let backend = "INJECTED" in
  match shape with
  | "compile" ->
      runtime backend (Compilation_failed {source = "<injected>"; log = "x"})
  | "codegen" ->
      E.Backend_error
        (E.Codegen
           {
             backend;
             error = Unsupported_construct {construct = "x"; reason = "x"};
           })
  | "launch" ->
      runtime backend (Context_error {operation = "launch"; reason = "x"})
  | "kernel_launch" ->
      runtime backend (Kernel_launch_failed {kernel_name = "k"; reason = "x"})
  | "module_load" ->
      runtime backend (Module_load_failed {size = 0; reason = "x"})
  | "alloc" ->
      runtime backend (Memory_allocation_failed {bytes = 0L; reason = "x"})
  | "feature" -> plugin backend (Feature_not_supported {feature = "x"; backend})
  | "failure" -> Failure "injected"
  | "rtc" -> Rtc_stub.Hiprtc_error "injected"
  | "unavailable" -> plugin backend (Backend_unavailable {reason = "x"})
  | "library" -> plugin backend (Library_not_found {library = "x"; paths = []})
  | "no_device" -> runtime backend (No_device_selected {operation = "x"})
  | "device_missing" ->
      runtime backend (Device_not_found {device_id = 0; max_devices = 0})
  | other -> failwith ("SPOC_TEST_INJECT_BACKEND_ERROR: unknown shape " ^ other)

(* Deterministic fault injection. SPOC_TEST_INJECT_BACKEND_ERROR names a shape
   (see [injected_exn]); [inject] raises it. Called inside a test's
   device-error scope, it lets the build prove the test fails on a compile
   error without needing a GPU that produces one. A no-op when unset. *)
let inject () =
  match Sys.getenv_opt "SPOC_TEST_INJECT_BACKEND_ERROR" with
  | None | Some "" -> ()
  | Some shape -> raise (injected_exn shape)

let fail_shapes =
  [
    "compile";
    "codegen";
    "launch";
    "kernel_launch";
    "module_load";
    "alloc";
    "feature";
    "failure";
    "rtc";
  ]

let skip_shapes = ["unavailable"; "library"; "no_device"; "device_missing"]

(* Self-check of the classification, run before any device: every regression
   shape must FAIL and only the not-there shapes may skip. *)
let self_check ~test_name =
  let expect want shape =
    if verdict (injected_exn shape) <> want then begin
      Printf.printf
        "%s: FAILED - injected %s failure is not classified as expected\n%!"
        test_name
        shape ;
      exit 1
    end
  in
  List.iter (expect `Fail) fail_shapes ;
  List.iter (expect `Skip) skip_shapes
