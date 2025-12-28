(******************************************************************************
 * E2E test for Sarek PPX - Sorting algorithms
 *
 * Tests bitonic sort and odd-even merge sort - parallel sorting algorithms
 * that work well on GPUs due to their regular communication patterns.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(** Bitonic sort step - one comparison/swap pass *)
let bitonic_sort_step_kernel =
  [%kernel
    fun (data : int32 vector) (j : int32) (k : int32) (n : int32) ->
      let i = thread_idx_x + (block_dim_x * block_idx_x) in
      if i < n then begin
        let ij = i lxor j in
        if ij > i then begin
          let di = data.(i) in
          let dij = data.(ij) in
          let ascending = i land k = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            data.(i) <- dij ;
            data.(ij) <- di
          end
        end
      end]

(** Block-level bitonic sort using shared memory with supersteps *)
let bitonic_sort_block_kernel =
  [%kernel
    fun (data : int32 vector) (n : int32) ->
      let%shared (shared : int32) = 256l in
      let tid = thread_idx_x in
      let gid = thread_idx_x + (block_dim_x * block_idx_x) in
      (* Load to shared memory *)
      let%superstep load =
        if gid < n then shared.(tid) <- data.(gid)
        else shared.(tid) <- 2147483647l
      in
      (* Bitonic sort stage k=2 *)
      let%superstep[@divergent] sort_k2 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 2l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=4, j=2 *)
      let%superstep[@divergent] sort_k4_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 4l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=4, j=1 *)
      let%superstep[@divergent] sort_k4_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 4l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=4 *)
      let%superstep[@divergent] sort_k8_j4 =
        let ij = tid lxor 4l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=2 *)
      let%superstep[@divergent] sort_k8_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=8, j=1 *)
      let%superstep[@divergent] sort_k8_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 8l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=8 *)
      let%superstep[@divergent] sort_k16_j8 =
        let ij = tid lxor 8l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=4 *)
      let%superstep[@divergent] sort_k16_j4 =
        let ij = tid lxor 4l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=2 *)
      let%superstep[@divergent] sort_k16_j2 =
        let ij = tid lxor 2l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Bitonic sort stage k=16, j=1 *)
      let%superstep[@divergent] sort_k16_j1 =
        let ij = tid lxor 1l in
        if ij > tid then begin
          let di = shared.(tid) in
          let dij = shared.(ij) in
          let ascending = tid land 16l = 0l in
          if (ascending && di > dij) || ((not ascending) && di < dij) then begin
            shared.(tid) <- dij ;
            shared.(ij) <- di
          end
        end
      in
      (* Write back *)
      if gid < n then data.(gid) <- shared.(tid)]

(** Odd-even transposition sort step *)
let odd_even_step_kernel =
  [%kernel
    fun (data : int32 vector) (phase : int32) (n : int32) ->
      let i = thread_idx_x + (block_dim_x * block_idx_x) in
      let idx = (2l * i) + phase in
      if idx + 1l < n then begin
        let a = data.(idx) in
        let b = data.(idx + 1l) in
        if a > b then begin
          data.(idx) <- b ;
          data.(idx + 1l) <- a
        end
      end]

(** Run global bitonic sort test *)
let run_bitonic_sort_global dev =
  (* Size must be power of 2 for bitonic sort *)
  let log2n = int_of_float (log (float_of_int cfg.size) /. log 2.0) in
  let n = 1 lsl log2n in
  let data = Vector.create Vector.int32 n in

  (* Initialize with random values *)
  Random.init 42 ;
  for i = 0 to n - 1 do
    Mem.set data i (Int32.of_int (Random.int 10000))
  done ;

  (* Sort using bitonic sort *)
  ignore (Sarek.Kirc.gen bitonic_sort_step_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  (* Outer loop: stages *)
  let k = ref 2 in
  while !k <= n do
    (* Inner loop: steps within stage *)
    let j = ref (!k / 2) in
    while !j > 0 do
      Sarek.Kirc.run
        bitonic_sort_step_kernel
        (data, !j, !k, n)
        (block, grid)
        0
        dev ;
      Devices.flush dev () ;
      j := !j / 2
    done ;
    k := !k * 2
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu data () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Mem.get data i > Mem.get data (i + 1) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run block-level bitonic sort test - sorts 16 elements *)
let run_bitonic_sort_block dev =
  let n = 16 in
  (* Fixed to 16 elements for unrolled supersteps *)
  let data = Vector.create Vector.int32 n in

  (* Initialize with random values *)
  Random.init 42 ;
  for i = 0 to n - 1 do
    Mem.set data i (Int32.of_int (Random.int 10000))
  done ;

  ignore (Sarek.Kirc.gen bitonic_sort_block_kernel dev) ;
  let block = {Kernel.blockX = n; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = 1; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run bitonic_sort_block_kernel (data, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu data () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Mem.get data i > Mem.get data (i + 1) then begin
          if !errors < 10 then
            Printf.printf
              "  Out of order at %d: %ld > %ld\n"
              i
              (Mem.get data i)
              (Mem.get data (i + 1)) ;
          incr errors
        end
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run odd-even transposition sort test *)
let run_odd_even_sort dev =
  let n = min 512 cfg.size in
  let data = Vector.create Vector.int32 n in

  (* Initialize with random values *)
  Random.init 42 ;
  for i = 0 to n - 1 do
    Mem.set data i (Int32.of_int (Random.int 10000))
  done ;

  ignore (Sarek.Kirc.gen odd_even_step_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = ((n / 2) + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = max 1 blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  (* n phases for complete sort *)
  for phase = 0 to n - 1 do
    let phase_mod = phase mod 2 in
    Sarek.Kirc.run odd_even_step_kernel (data, phase_mod, n) (block, grid) 0 dev ;
    Devices.flush dev ()
  done ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu data () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 2 do
        if Mem.get data i > Mem.get data (i + 1) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_sort" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.use_native_parallel <- c.use_native_parallel ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.verify <- c.verify ;
  cfg.size <- c.size ;
  cfg.block_size <- c.block_size ;

  let devs = Devices.init () in
  if Array.length devs = 0 then begin
    print_endline "No GPU devices found" ;
    exit 1
  end ;
  Test_helpers.print_devices devs ;

  if cfg.benchmark_all then begin
    Test_helpers.benchmark_all
      devs
      run_bitonic_sort_global
      "Bitonic sort (global)" ;
    Test_helpers.benchmark_all
      devs
      run_bitonic_sort_block
      "Bitonic sort (block)" ;
    Test_helpers.benchmark_all devs run_odd_even_sort "Odd-even sort"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing sorting algorithms\n%!" ;

    Printf.printf "\nBitonic sort (global):\n%!" ;
    let time_ms, ok = run_bitonic_sort_global dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nBitonic sort (block-level):\n%!" ;
    let time_ms, ok = run_bitonic_sort_block dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    Printf.printf "\nOdd-even transposition sort:\n%!" ;
    let time_ms, ok = run_odd_even_sort dev in
    Printf.printf
      "  Time: %.4f ms, %s\n%!"
      time_ms
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nSort tests PASSED"
  end
