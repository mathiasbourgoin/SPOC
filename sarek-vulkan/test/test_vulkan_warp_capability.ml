(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The Vulkan claims about subgroup shuffles, with no device and no driver call:
   the capability bits (VkPhysicalDeviceSubgroupProperties), the extension lines
   the GLSL declares, and the compile environment the SPIR-V needs. *)

open Alcotest

let supported = Sarek_vulkan.Vulkan_api_device.subgroup_shuffle_supported

let basic = 0x1

let shuffle = 0x10

let shuffle_relative = 0x20

let compute = 0x20

let vertex = 0x1

let test_bits () =
  check
    bool
    "basic + shuffle, compute"
    true
    (supported ~operations:(basic lor shuffle) ~stages:compute) ;
  check
    bool
    "all operations, all stages"
    true
    (supported ~operations:0xFF ~stages:0x7F) ;
  check
    bool
    "shuffle_relative is NOT required (up/down/xor are index shuffles)"
    true
    (supported ~operations:(basic lor shuffle) ~stages:compute) ;
  check
    bool
    "no shuffle bit"
    false
    (supported ~operations:(basic lor shuffle_relative) ~stages:compute) ;
  check
    bool
    "no basic bit"
    false
    (supported ~operations:shuffle ~stages:compute) ;
  check
    bool
    "compute stage missing"
    false
    (supported ~operations:(basic lor shuffle) ~stages:vertex) ;
  check
    bool
    "unwritten struct (Vulkan 1.0)"
    false
    (supported ~operations:0 ~stages:0)

let kernel ~name =
  let open Sarek_ir_types in
  let dst =
    {var_name = "d"; var_id = 0; var_type = TVec TInt32; var_mutable = false}
  in
  {
    default_kernel with
    kern_name = "k";
    kern_params =
      [DParam (dst, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body =
      SAssign
        ( LArrayElem ("d", EConst (CInt32 0l)),
          EIntrinsic
            ( [],
              name,
              if name = "lane_id" then []
              else [EConst (CInt32 1l); EConst (CInt32 1l)] ) );
  }

let has src needle =
  let n = String.length needle and m = String.length src in
  let rec go i = i + n <= m && (String.sub src i n = needle || go (i + 1)) in
  go 0

let test_extensions () =
  let shuffle_src =
    Sarek_vulkan.Sarek_ir_glsl.generate
      (kernel ~name:(Sarek_warp.ir_name Sarek_warp.Xor Sarek_warp.I32))
  in
  check bool "basic" true (has shuffle_src "GL_KHR_shader_subgroup_basic") ;
  check bool "shuffle" true (has shuffle_src "GL_KHR_shader_subgroup_shuffle") ;
  check
    bool
    "no relative-shuffle extension"
    false
    (has shuffle_src "shuffle_relative") ;
  let lane_src = Sarek_vulkan.Sarek_ir_glsl.generate (kernel ~name:"lane_id") in
  check
    bool
    "lane_id: basic only"
    true
    (has lane_src "GL_KHR_shader_subgroup_basic"
    && not (has lane_src "GL_KHR_shader_subgroup_shuffle")) ;
  let plain =
    Sarek_vulkan.Sarek_ir_glsl.generate (kernel ~name:"global_thread_id")
  in
  check
    bool
    "no warp operation: no subgroup extension"
    false
    (has plain "GL_KHR_shader_subgroup")

let test_compile_environment () =
  let needs = Sarek_vulkan.Vulkan_api_base.needs_vulkan_1_1 in
  check
    bool
    "shuffle shader"
    true
    (needs
       (Sarek_vulkan.Sarek_ir_glsl.generate
          (kernel ~name:(Sarek_warp.ir_name Sarek_warp.Idx Sarek_warp.F32)))) ;
  check
    bool
    "other shaders keep the default environment"
    false
    (needs
       (Sarek_vulkan.Sarek_ir_glsl.generate (kernel ~name:"global_thread_id")))

let () =
  run
    "vulkan warp capability"
    [
      ( "subgroup shuffle",
        [
          test_case "capability bits" `Quick test_bits;
          test_case "extension lines" `Quick test_extensions;
          test_case "compile environment" `Quick test_compile_environment;
        ] );
    ]
