(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The hidden CUDA/C backend (NVRTC, nvcuda::wmma), registered on request. *)
let register () =
  if not (Sarek_cuda.Cuda_shared.is_disabled ()) then
    Sarek_cuda.Cuda_c_plugin.register ()
