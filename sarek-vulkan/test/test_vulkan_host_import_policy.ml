(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host-memory import on Vulkan is explicit opt-in and refused on AMD drivers
 * (Sarek #503: amdgpu_hmm_invalidate_gfx oops after VK_EXT_external_memory_host
 * imports). The decision is checked as a pure function, and on a real device
 * (skipped without one) the default is: no import, both entry points refuse
 * with Feature_not_supported before any host page is handed to the driver. The
 * dune rule clears the opt-in variables, so this never imports.
 ******************************************************************************)

open Sarek_vulkan
module Policy = Vulkan_host_import_policy
module Device = Vulkan_api_device
module Host = Vulkan_api_host
module BE = Sarek_backend_error.Backend_error

let radv = 3

let nvidia_vendor = 0x10de

let amd_vendor = 0x1002

let decide ?(opted_in = true) ?(amdgpu_override = false) ~vendor_id ~driver_id
    () =
  Policy.decide ~opted_in ~amdgpu_override ~vendor_id ~driver_id

let decision =
  Alcotest.testable
    (fun fmt d ->
      Format.pp_print_string
        fmt
        (match d with
        | Policy.Allowed -> "Allowed"
        | Not_opted_in -> "Not_opted_in"
        | Amdgpu_refused -> "Amdgpu_refused"))
    ( = )

let test_default_is_off () =
  List.iter
    (fun (vendor_id, driver_id) ->
      Alcotest.check
        decision
        "not opted in"
        Policy.Not_opted_in
        (decide ~opted_in:false ~vendor_id ~driver_id ()))
    [(nvidia_vendor, 4); (amd_vendor, radv); (0, 0)]

let test_amd_refused_even_opted_in () =
  Alcotest.check
    decision
    "RADV by driver id"
    Policy.Amdgpu_refused
    (decide ~vendor_id:0 ~driver_id:radv ()) ;
  Alcotest.check
    decision
    "AMD by vendor id"
    Policy.Amdgpu_refused
    (decide ~vendor_id:amd_vendor ~driver_id:0 ())

let test_other_drivers_allowed_when_opted_in () =
  Alcotest.check
    decision
    "non-AMD"
    Policy.Allowed
    (decide ~vendor_id:nvidia_vendor ~driver_id:4 ())

let test_amd_override () =
  Alcotest.check
    decision
    "explicit override"
    Policy.Allowed
    (decide ~amdgpu_override:true ~vendor_id:amd_vendor ~driver_id:radv ()) ;
  Alcotest.check
    decision
    "override without opt-in stays off"
    Policy.Not_opted_in
    (decide
       ~opted_in:false
       ~amdgpu_override:true
       ~vendor_id:amd_vendor
       ~driver_id:radv
       ())

let test_reason_text () =
  Alcotest.(check bool)
    "allowed has none"
    true
    (Policy.refusal_reason Allowed = None) ;
  List.iter
    (fun d ->
      Alcotest.(check bool)
        "refusals carry a reason"
        true
        (Policy.refusal_reason d <> None))
    [Policy.Not_opted_in; Policy.Amdgpu_refused]

let expect_refused what f =
  match f () with
  | _ -> Alcotest.failf "%s was not refused" what
  | exception
      BE.Backend_error (BE.Plugin {error = BE.Feature_not_supported _; _}) ->
      ()

let test_device_default_refuses () =
  if not (Vulkan_api.is_available ()) then Alcotest.skip () ;
  Device.init () ;
  let device = Device.get 0 in
  Alcotest.check
    decision
    "default policy"
    Policy.Not_opted_in
    device.Device.host_import_policy ;
  Alcotest.(check bool)
    "no import granule, the extension was not enabled"
    true
    (device.Device.host_import_alignment = None
    && device.Device.host_pointer_properties_fn = None) ;
  expect_refused "alloc_pinned" (fun () ->
      Host.alloc_pinned device ~byte_size:4096) ;
  let page = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8192 in
  let addr =
    Ctypes.raw_address_of_ptr (Spoc_core.Memory.borrow_bigarray_ptr page)
  in
  expect_refused "register" (fun () ->
      Host.register device addr ~byte_size:8192)

(* Run with SPOC_VULKAN_HOST_IMPORT=1 (argument [optin]): an AMD device must
   still refuse, with the extension off; any other device may import. Nothing is
   imported here. *)
let test_device_optin () =
  if not (Vulkan_api.is_available ()) then Alcotest.skip () ;
  Device.init () ;
  let device = Device.get 0 in
  if Policy.is_amd ~vendor_id:0 ~driver_id:device.Device.driver_id then begin
    Alcotest.check
      decision
      "AMD refused although opted in"
      Policy.Amdgpu_refused
      device.Device.host_import_policy ;
    Alcotest.(check bool)
      "extension off"
      true
      (device.Device.host_import_alignment = None
      && device.Device.host_pointer_properties_fn = None) ;
    expect_refused "alloc_pinned" (fun () ->
        Host.alloc_pinned device ~byte_size:4096)
  end
  else
    Alcotest.(check bool)
      "not refused by policy"
      true
      (device.Device.host_import_policy <> Policy.Amdgpu_refused)

let () =
  if Array.length Sys.argv > 1 && Sys.argv.(1) = "optin" then
    Alcotest.run
      ~argv:[|Sys.argv.(0)|]
      "vulkan_host_import_policy_optin"
      [
        ( "opt-in",
          [Alcotest.test_case "AMD still refused" `Quick test_device_optin] );
      ]
  else
    Alcotest.run
      "vulkan_host_import_policy"
      [
        ( "policy",
          [
            Alcotest.test_case "off by default" `Quick test_default_is_off;
            Alcotest.test_case
              "AMD refused even when opted in"
              `Quick
              test_amd_refused_even_opted_in;
            Alcotest.test_case
              "other drivers allowed when opted in"
              `Quick
              test_other_drivers_allowed_when_opted_in;
            Alcotest.test_case "AMD override" `Quick test_amd_override;
            Alcotest.test_case "refusal reasons" `Quick test_reason_text;
          ] );
        ( "device",
          [
            Alcotest.test_case
              "default refuses both entry points"
              `Quick
              test_device_default_refuses;
          ] );
      ]
