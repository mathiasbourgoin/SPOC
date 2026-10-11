(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The hidden CUDA/C backend (NVRTC, nvcuda::wmma), registered on request. *)
let register () =
  if not (Sarek_cuda.Cuda_shared.is_disabled ()) then
    Sarek_cuda.Cuda_c_plugin.register ()

(* CUDA C source of an IR kernel, and its PTX for sm_80 through NVRTC; [None]
   when NVRTC or a CUDA include directory is missing. *)
let source_of (ir : Sarek_ir_types.kernel) =
  Sarek_cuda.Sarek_ir_cuda.generate_with_types
    ~types:ir.Sarek_ir_types.kern_types
    ir

let is_zluda = Sarek_cuda.Cuda_coopmat.is_zluda

let probe ~device_name ~compute_capability ~warp_size =
  Some
    (Sarek_cuda.Cuda_coopmat.support
       ~device_name
       ~compute_capability
       ~warp_size)

let ptx_of ~name source =
  if
    Sarek_cuda.Cuda_nvrtc.is_available ()
    && Lazy.force Sarek_cuda.Cuda_nvrtc.cuda_include_paths <> []
  then
    Some (Sarek_cuda.Cuda_nvrtc.compile_to_ptx ~name ~arch:"compute_80" source)
  else None
