(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Shared classification of a failure raised while a test drives a device.

   Rule: only "the backend/feature is not there" may skip a test. Anything that
   says the source we generated was wrong, the device compiler or module loader
   rejected it, or the launch failed is a regression and must FAIL. Without this
   a CI runner whose first device is CPU OpenCL turns invalid generated OpenCL C
   into a green "SKIPPED (backend error)" (docs review D-40, D-41). *)

module E = Sarek_backend_error.Backend_error

(* [Hiprtc_error] / [Nvrtc_error] are raised by the runtime-compilation
   bindings and reach the caller without being wrapped in [Backend_error]
   (runtime review R-40). A failed hiprtc/nvrtc compile is a compile failure.
   Matched by constructor name so that this library need not link the optional
   HIP and CUDA plugins. *)
let is_rtc_compile_error exn =
  let slot = Printexc.exn_slot_name exn in
  let ends_with suffix =
    let ls = String.length slot and lx = String.length suffix in
    ls >= lx && String.sub slot (ls - lx) lx = suffix
  in
  ends_with "Hiprtc_error" || ends_with "Nvrtc_error"

(* A failure that must FAIL the test: codegen raised, the device compiler or
   loader rejected what we emitted, the launch failed ([Context_error] is what a
   failed cuLaunchKernel/hipModuleLaunchKernel becomes), or a runtime-compile
   binding raised. An unavailable backend or unsupported feature is not here. *)
let rejected_source = function
  | E.Backend_error
      ( Codegen _
      | Runtime
          {
            error =
              ( Compilation_failed _ | Module_load_failed _ | Context_error _
              | Kernel_launch_failed _ );
            _;
          } ) ->
      true
  | exn -> is_rtc_compile_error exn

(* A [Backend_error] that only says the device/backend cannot do this here: the
   backend or a feature is not there, no such device, or the (shared) device is
   out of memory. *)
let is_unavailable = function
  | E.Backend_error
      ( Plugin _
      | Runtime
          {
            error =
              ( No_device_selected _ | Device_not_found _
              | Memory_allocation_failed _ );
            _;
          } ) ->
      true
  | _ -> false

(* [`Skip] only for an unavailable backend; every other exception is a
   [`Fail]. *)
let verdict exn =
  if is_unavailable exn && not (rejected_source exn) then `Skip else `Fail

(* Stand-in for the runtime-compile binding's exception, so the injection can
   exercise the by-name match without linking the HIP plugin. *)
module Rtc_stub = struct
  exception Hiprtc_error of string
end

(* The exception a given injection shape raises. *)
let injected_exn shape =
  let backend = "INJECTED" in
  match shape with
  | "compile" ->
      E.Backend_error
        (E.Runtime
           {
             backend;
             error =
               Compilation_failed {source = "<injected>"; log = "injected"};
           })
  | "codegen" ->
      E.Backend_error
        (E.Codegen
           {
             backend;
             error =
               Unsupported_construct
                 {construct = "injected"; reason = "injected"};
           })
  | "launch" ->
      E.Backend_error
        (E.Runtime
           {
             backend;
             error = Context_error {operation = "launch"; reason = "injected"};
           })
  | "rtc" -> Rtc_stub.Hiprtc_error "injected"
  | "unavailable" ->
      E.Backend_error
        (E.Plugin {backend; error = Backend_unavailable {reason = "injected"}})
  | other -> failwith ("SPOC_TEST_INJECT_BACKEND_ERROR: unknown shape " ^ other)

(* Deterministic fault injection. SPOC_TEST_INJECT_BACKEND_ERROR (compile |
   codegen | launch | rtc | unavailable) makes this raise the matching failure.
   Called inside a test's device-error scope, it lets the build prove the test
   fails on a compile error without needing a GPU that produces one. A no-op
   when the variable is unset. *)
let inject () =
  match Sys.getenv_opt "SPOC_TEST_INJECT_BACKEND_ERROR" with
  | None | Some "" -> ()
  | Some shape -> raise (injected_exn shape)

(* Self-check of the classification, run before any device: a regression must
   not be classified as skippable, and an unavailable backend must not be
   classified as a regression. *)
let self_check ~test_name =
  let expect shape want =
    if verdict (injected_exn shape) <> want then begin
      Printf.printf
        "%s: FAILED - injected %s failure is not classified as expected\n%!"
        test_name
        shape ;
      exit 1
    end
  in
  List.iter (fun s -> expect s `Fail) ["compile"; "codegen"; "launch"; "rtc"] ;
  expect "unavailable" `Skip
