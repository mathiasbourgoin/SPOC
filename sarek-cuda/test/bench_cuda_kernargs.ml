(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Host-side cost of one [Kernel.launch] with 3 and 10 int32 arguments on an
    empty PTX kernel (specs/cheap-repeated-launches.md US-4). Prints one line
    per argument count: [args=N us_per_launch=X]. Hand-run, interleaved with a
    build of the previous marshalling: not part of [dune test]. Skips without a
    CUDA driver. *)

open Sarek_cuda
module K = Cuda_api.Kernel

let launches = 4000

let empty_kernel_ptx n =
  let params =
    String.concat
      ",\n"
      (List.init n (fun i -> Printf.sprintf "    .param .u32 p%d" i))
  in
  Printf.sprintf
    ".version 8.0\n\
     .target sm_86\n\
     .address_size 64\n\n\
     .entry empty_args(\n\
     %s\n\
     )\n\
     {\n\
    \    ret;\n\
     }\n"
    params

let time_launches dev n =
  let kernel =
    K.load_from_ptx dev ~name:"empty_args" ~ptx:(empty_kernel_ptx n)
  in
  let launch () =
    K.launch
      kernel
      ~args:(List.init n (fun i -> K.ArgInt32 (Int32.of_int i)))
      ~grid:(1, 1, 1)
      ~block:(1, 1, 1)
      ~shared_mem:0
      ~stream:None
  in
  for _ = 1 to 200 do
    launch ()
  done ;
  Cuda_api.Device.synchronize dev ;
  let t0 = Unix.gettimeofday () in
  for _ = 1 to launches do
    launch ()
  done ;
  let t1 = Unix.gettimeofday () in
  Cuda_api.Device.synchronize dev ;
  (t1 -. t0) *. 1e6 /. float_of_int launches

let () =
  if not (Cuda_api.is_driver_available ()) then
    print_endline "bench_cuda_kernargs SKIPPED (no CUDA)"
  else begin
    Cuda_api.Device.init () ;
    let dev = Cuda_api.Device.get 0 in
    List.iter
      (fun n ->
        Printf.printf "args=%d us_per_launch=%.3f\n%!" n (time_launches dev n))
      [3; 10]
  end
