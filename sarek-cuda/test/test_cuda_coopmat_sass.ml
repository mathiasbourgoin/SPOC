(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** The CUDA arm of the cooperative-matrix lowering reaches the tensor cores
    (specs/coopmat-hip-wmma.md).

    generated CUDA -> NVRTC (compute_80) -> ptxas sm_80 -> nvdisasm -> SASS,
    then: an f16 kernel contains [HMMA] and an 8-bit kernel contains [IMMA].
    Host tools only; no device. This is what "compile-checked" means for the
    CUDA arm: no NVIDIA GPU was available, so nothing here says the results are
    right — only that the instruction the lowering exists to reach is the one
    the toolchain emitted. Skips cleanly when libnvrtc, a CUDA include
    directory, ptxas or nvdisasm is missing; ci/assert-toolchain.sh is what
    keeps that skip from returning silently in CI. *)

open Sarek_ir_types
module C = Sarek_coopmat

let tile name a b acc =
  let cfg =
    {
      C.cfg_shape = {C.m = 16; n = 16; k = 16};
      cfg_a = a;
      cfg_b = b;
      cfg_c = acc;
      cfg_result = acc;
      cfg_saturating = false;
      cfg_scope = C.Subgroup;
    }
  in
  let fa, fb, fc, _ = C.fragments_of_config cfg in
  let elt = function
    | C.Float16 -> TFloat16
    | C.BFloat16 | C.Float32 -> TFloat32
    | C.Uint8 | C.Sint8 -> TUint8
    | C.Uint32 | C.Sint32 -> TInt32
  in
  let param n e =
    DParam
      ( {var_name = n; var_id = 0; var_type = TVec e; var_mutable = false},
        Some {arr_elttype = e; arr_memspace = Global} )
  in
  let i = EConst (CInt32 0l) and s = EConst (CInt32 16l) in
  let load dst frag src =
    SCoopmat
      (CM_load
         {
           dst;
           frag;
           src;
           index = i;
           stride = s;
           layout = Sarek_coopmat_types.Row_major;
         })
  in
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [
        param "a" (elt a);
        param "b" (elt b);
        param "c" (elt acc);
        param "d" (elt acc);
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
                 frag = fc;
                 dst = "d";
                 index = i;
                 stride = s;
                 layout = Sarek_coopmat_types.Row_major;
               });
        ];
  }

let on_path tool =
  Unix.system (Printf.sprintf "command -v %s >/dev/null 2>&1" tool)
  = Unix.WEXITED 0

let skip_reason () =
  if not (Sarek_cuda.Cuda_nvrtc.is_available ()) then
    Some "libnvrtc not loadable"
  else if Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths = [] then
    Some "no CUDA include directory"
  else if not (on_path "ptxas") then Some "ptxas not on PATH"
  else if not (on_path "nvdisasm") then Some "nvdisasm not on PATH"
  else None

let read_file path = In_channel.with_open_bin path In_channel.input_all

let sass_of_ptx ptx =
  let base = Filename.temp_file "sarek_coopmat_sass_" "" in
  let ptx_f = base ^ ".ptx" and cubin = base ^ ".cubin" in
  let sass = base ^ ".sass" in
  Fun.protect
    ~finally:(fun () ->
      List.iter
        (fun f -> try Sys.remove f with _ -> ())
        [base; ptx_f; cubin; sass])
    (fun () ->
      Out_channel.with_open_bin ptx_f (fun oc -> output_string oc ptx) ;
      let run cmd =
        if Sys.command cmd <> 0 then Alcotest.failf "command failed: %s" cmd
      in
      run
        (Printf.sprintf
           "ptxas -arch=sm_80 -o %s %s"
           (Filename.quote cubin)
           (Filename.quote ptx_f)) ;
      run
        (Printf.sprintf
           "nvdisasm -c %s > %s"
           (Filename.quote cubin)
           (Filename.quote sass)) ;
      read_file sass)

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  go 0

let case name k opcode =
  Alcotest.test_case name `Quick (fun () ->
      let src = Sarek_codegen.Sarek_ir_cuda.generate k in
      match skip_reason () with
      | Some why -> Printf.printf "  SKIP %s: %s\n%!" name why
      | None ->
          let ptx =
            Sarek_cuda.Cuda_nvrtc.compile_to_ptx ~name ~arch:"compute_80" src
          in
          let sass = sass_of_ptx ptx in
          if not (contains sass opcode) then
            Alcotest.failf "%s: no %s in the sm_80 SASS:\n%s" name opcode sass ;
          Printf.printf "  %s: %s present in sm_80 SASS\n%!" name opcode)

let () =
  Alcotest.run
    "cuda_coopmat_sass"
    [
      ( "tensor-core-opcode",
        [
          case "f16_f32" (tile "cm_f16" C.Float16 C.Float16 C.Float32) "HMMA";
          case "u8_s32" (tile "cm_u8" C.Uint8 C.Uint8 C.Sint32) "IMMA";
          case "s8_s32" (tile "cm_s8" C.Sint8 C.Sint8 C.Sint32) "IMMA";
        ] );
    ]
