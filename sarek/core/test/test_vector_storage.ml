(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for Vector_storage module
 *
 * Covers creation helpers, copy/sub_vector/partition host logic, and subvector
 * metadata queries. These tests are CPU-only.
 ******************************************************************************)

open Spoc_core.Vector

let test_copy_host_only () =
  let v = create_float32 4 in
  set v 0 1.0 ;
  set v 1 2.0 ;
  let v2 = copy v in
  assert (length v2 = 4) ;
  assert (get v2 1 = 2.0) ;
  set v 1 9.0 ;
  assert (get v2 1 = 2.0) ;
  print_endline "  copy_host_only: OK"

let test_sub_vector_metadata () =
  let v = create_int32 10 in
  let sub = sub_vector v ~start:2 ~len:5 () in
  assert (is_sub sub) ;
  assert (parent_id sub = Some (id v)) ;
  assert (sub_start sub = Some 2) ;
  assert (depth sub = 1) ;
  print_endline "  sub_vector metadata: OK"

let test_partition_host () =
  let v = create_int64 9 in
  let devs =
    Array.init 3 (fun i ->
        {
          Spoc_core.Device.id = i;
          backend_id = i;
          name = "Dummy";
          framework = "Native";
          capabilities =
            {
              max_threads_per_block = 256;
              max_block_dims = (256, 256, 64);
              max_grid_dims = (65535, 65535, 65535);
              shared_mem_per_block = 16384;
              total_global_mem = 0L;
              compute_capability = (0, 0);
              supports_fp64 = true;
              supports_atomics = true;
              warp_size = 32;
              max_registers_per_block = 16384;
              clock_rate_khz = 0;
              multiprocessor_count = 1;
              is_cpu = true;
            };
        })
  in
  let parts = partition v devs in
  assert (Array.length parts = 3) ;
  assert (length parts.(0) = 3) ;
  assert (sub_start parts.(1) = Some 3) ;
  assert (depth parts.(2) = 1) ;
  print_endline "  partition_host: OK"

let () =
  print_endline "Vector_storage tests:" ;
  test_copy_host_only () ;
  test_sub_vector_metadata () ;
  test_partition_host () ;
  print_endline "All Vector_storage tests passed!"
