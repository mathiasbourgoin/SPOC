(******************************************************************************
 * E2E test for Sarek PPX - Bitwise operations
 *
 * Tests bitwise AND, OR, XOR, NOT, shifts, and bit manipulation patterns.
 * These are essential for cryptography, compression, and low-level algorithms.
 ******************************************************************************)

open Spoc

let cfg = Test_helpers.default_config ()

(* ========== Pure OCaml baselines ========== *)

let ocaml_bitwise_basic a b and_out or_out xor_out n =
  for i = 0 to n - 1 do
    and_out.(i) <- Int32.logand a.(i) b.(i) ;
    or_out.(i) <- Int32.logor a.(i) b.(i) ;
    xor_out.(i) <- Int32.logxor a.(i) b.(i)
  done

let ocaml_shift input shift_amt left_out right_out n =
  for i = 0 to n - 1 do
    let s = Int32.to_int shift_amt.(i) in
    left_out.(i) <- Int32.shift_left input.(i) s ;
    right_out.(i) <- Int32.shift_right_logical input.(i) s
  done

let count_bits x =
  let rec loop acc v =
    if v = 0l then acc
    else loop (Int32.add acc (Int32.logand v 1l)) (Int32.shift_right_logical v 1)
  in
  loop 0l x

let ocaml_popcount input output n =
  for i = 0 to n - 1 do
    output.(i) <- count_bits input.(i)
  done

let ocaml_gray_code input to_gray from_gray n =
  for i = 0 to n - 1 do
    let x = input.(i) in
    let gray = Int32.logxor x (Int32.shift_right_logical x 1) in
    to_gray.(i) <- gray ;
    (* Gray code to binary *)
    let g = gray in
    let b = ref g in
    let mask = ref (Int32.shift_right_logical g 1) in
    while !mask <> 0l do
      b := Int32.logxor !b !mask ;
      mask := Int32.shift_right_logical !mask 1
    done ;
    from_gray.(i) <- !b
  done

(* ========== Shared test data ========== *)

let input_a = ref [||]
let input_b = ref [||]
let expected_and = ref [||]
let expected_or = ref [||]
let expected_xor = ref [||]

let input_shift = ref [||]
let input_shift_amt = ref [||]
let expected_left = ref [||]
let expected_right = ref [||]

let input_popcount = ref [||]
let expected_popcount = ref [||]

let input_gray = ref [||]
let expected_to_gray = ref [||]
let expected_from_gray = ref [||]

let init_bitwise_data () =
  let n = cfg.size in
  let a = Array.init n (fun i -> Int32.of_int (i * 17)) in
  let b = Array.init n (fun i -> Int32.of_int (i * 13)) in
  let and_out = Array.make n 0l in
  let or_out = Array.make n 0l in
  let xor_out = Array.make n 0l in
  input_a := a ;
  input_b := b ;
  expected_and := and_out ;
  expected_or := or_out ;
  expected_xor := xor_out ;
  let t0 = Unix.gettimeofday () in
  ocaml_bitwise_basic a b and_out or_out xor_out n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_shift_data () =
  let n = cfg.size in
  let inp = Array.make n (Int32.of_int 0xFF00FF) in
  let amt = Array.init n (fun i -> Int32.of_int (i mod 16)) in
  let left = Array.make n 0l in
  let right = Array.make n 0l in
  input_shift := inp ;
  input_shift_amt := amt ;
  expected_left := left ;
  expected_right := right ;
  let t0 = Unix.gettimeofday () in
  ocaml_shift inp amt left right n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_popcount_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> Int32.of_int i) in
  let out = Array.make n 0l in
  input_popcount := inp ;
  expected_popcount := out ;
  let t0 = Unix.gettimeofday () in
  ocaml_popcount inp out n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

let init_gray_data () =
  let n = cfg.size in
  let inp = Array.init n (fun i -> Int32.of_int i) in
  let to_g = Array.make n 0l in
  let from_g = Array.make n 0l in
  input_gray := inp ;
  expected_to_gray := to_g ;
  expected_from_gray := from_g ;
  let t0 = Unix.gettimeofday () in
  ocaml_gray_code inp to_g from_g n ;
  let t1 = Unix.gettimeofday () in
  ((t1 -. t0) *. 1000.0, true)

(* ========== Sarek kernels ========== *)

(** Basic bitwise operations kernel *)
let bitwise_basic_kernel =
  [%kernel
    fun (a : int32 vector)
        (b : int32 vector)
        (and_out : int32 vector)
        (or_out : int32 vector)
        (xor_out : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = a.(tid) in
        let y = b.(tid) in
        and_out.(tid) <- x land y ;
        or_out.(tid) <- x lor y ;
        xor_out.(tid) <- x lxor y
      end]

(** Shift operations kernel *)
let shift_kernel =
  [%kernel
    fun (input : int32 vector)
        (shift_amt : int32 vector)
        (left_out : int32 vector)
        (right_out : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        let s = shift_amt.(tid) in
        left_out.(tid) <- x lsl s ;
        right_out.(tid) <- x lsr s
      end]

(** Bit counting kernel - count number of 1 bits (popcount) *)
let popcount_kernel =
  [%kernel
    fun (input : int32 vector) (output : int32 vector) (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = mut input.(tid) in
        let count = mut 0l in
        while x <> 0l do
          count := count + (x land 1l) ;
          x := x lsr 1l
        done ;
        output.(tid) <- count
      end]

(** Gray code conversion kernel *)
let gray_code_kernel =
  [%kernel
    fun (input : int32 vector)
        (to_gray : int32 vector)
        (from_gray : int32 vector)
        (n : int32) ->
      let tid = thread_idx_x + (block_dim_x * block_idx_x) in
      if tid < n then begin
        let x = input.(tid) in
        (* Binary to Gray code: x XOR (x >> 1) *)
        to_gray.(tid) <- x lxor (x lsr 1l) ;
        (* Gray code to binary (for verification) *)
        let g = x lxor (x lsr 1l) in
        let b = mut g in
        let mask = mut (g lsr 1l) in
        while mask <> 0l do
          b := b lxor mask ;
          mask := mask lsr 1l
        done ;
        from_gray.(tid) <- b
      end]

(* ========== Device test runners ========== *)

(** Run basic bitwise test *)
let run_bitwise_basic_test dev =
  let n = cfg.size in
  let inp_a = !input_a in
  let inp_b = !input_b in
  let exp_and = !expected_and in
  let exp_or = !expected_or in
  let exp_xor = !expected_xor in

  let a = Vector.create Vector.int32 n in
  let b = Vector.create Vector.int32 n in
  let and_out = Vector.create Vector.int32 n in
  let or_out = Vector.create Vector.int32 n in
  let xor_out = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set a i inp_a.(i) ;
    Mem.set b i inp_b.(i) ;
    Mem.set and_out i 0l ;
    Mem.set or_out i 0l ;
    Mem.set xor_out i 0l
  done ;

  ignore (Sarek.Kirc.gen bitwise_basic_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    bitwise_basic_kernel
    (a, b, and_out, or_out, xor_out, n)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu and_out () ;
      Mem.to_cpu or_out () ;
      Mem.to_cpu xor_out () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if
          Mem.get and_out i <> exp_and.(i)
          || Mem.get or_out i <> exp_or.(i)
          || Mem.get xor_out i <> exp_xor.(i)
        then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run shift test *)
let run_shift_test dev =
  let n = cfg.size in
  let inp = !input_shift in
  let amt = !input_shift_amt in
  let exp_left = !expected_left in
  let exp_right = !expected_right in

  let input = Vector.create Vector.int32 n in
  let shift_amt = Vector.create Vector.int32 n in
  let left_out = Vector.create Vector.int32 n in
  let right_out = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set shift_amt i amt.(i) ;
    Mem.set left_out i 0l ;
    Mem.set right_out i 0l
  done ;

  ignore (Sarek.Kirc.gen shift_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run
    shift_kernel
    (input, shift_amt, left_out, right_out, n)
    (block, grid)
    0
    dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu left_out () ;
      Mem.to_cpu right_out () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if Mem.get left_out i <> exp_left.(i) || Mem.get right_out i <> exp_right.(i)
        then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run popcount test *)
let run_popcount_test dev =
  let n = cfg.size in
  let inp = !input_popcount in
  let exp = !expected_popcount in

  let input = Vector.create Vector.int32 n in
  let output = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set output i 0l
  done ;

  ignore (Sarek.Kirc.gen popcount_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run popcount_kernel (input, output, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu output () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if Mem.get output i <> exp.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

(** Run Gray code test *)
let run_gray_code_test dev =
  let n = cfg.size in
  let inp = !input_gray in
  let exp_to = !expected_to_gray in
  let exp_from = !expected_from_gray in

  let input = Vector.create Vector.int32 n in
  let to_gray = Vector.create Vector.int32 n in
  let from_gray = Vector.create Vector.int32 n in

  for i = 0 to n - 1 do
    Mem.set input i inp.(i) ;
    Mem.set to_gray i 0l ;
    Mem.set from_gray i 0l
  done ;

  ignore (Sarek.Kirc.gen gray_code_kernel dev) ;
  let block_size = Test_helpers.get_block_size cfg dev in
  let blocks = (n + block_size - 1) / block_size in
  let block = {Kernel.blockX = block_size; blockY = 1; blockZ = 1} in
  let grid = {Kernel.gridX = blocks; gridY = 1; gridZ = 1} in

  let t0 = Unix.gettimeofday () in
  Sarek.Kirc.run gray_code_kernel (input, to_gray, from_gray, n) (block, grid) 0 dev ;
  Devices.flush dev () ;
  let t1 = Unix.gettimeofday () in
  let time_ms = (t1 -. t0) *. 1000.0 in

  let ok =
    if cfg.verify then begin
      Mem.to_cpu to_gray () ;
      Mem.to_cpu from_gray () ;
      Devices.flush dev () ;
      let errors = ref 0 in
      for i = 0 to n - 1 do
        if Mem.get to_gray i <> exp_to.(i) then incr errors ;
        if Mem.get from_gray i <> exp_from.(i) then incr errors
      done ;
      !errors = 0
    end
    else true
  in
  (time_ms, ok)

let () =
  let c = Test_helpers.parse_args "test_bitwise_ops" in
  cfg.dev_id <- c.dev_id ;
  cfg.use_interpreter <- c.use_interpreter ;
  cfg.use_native <- c.use_native ;
  cfg.benchmark_all <- c.benchmark_all ;
  cfg.benchmark_devices <- c.benchmark_devices ;
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
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_bitwise_data
      run_bitwise_basic_test
      "Bitwise AND/OR/XOR" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_shift_data
      run_shift_test
      "Bit shifts" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_popcount_data
      run_popcount_test
      "Popcount" ;
    Test_helpers.benchmark_with_baseline
      ~device_ids:cfg.benchmark_devices
      devs
      ~baseline:init_gray_data
      run_gray_code_test
      "Gray code"
  end
  else begin
    let dev = Test_helpers.get_device cfg devs in
    Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name ;
    Printf.printf "Testing bitwise operations with size=%d\n%!" cfg.size ;

    let baseline_ms, _ = init_bitwise_data () in
    Printf.printf "\nOCaml baseline (bitwise): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nBitwise AND/OR/XOR:\n%!" ;
    let time_ms, ok = run_bitwise_basic_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_shift_data () in
    Printf.printf "\nOCaml baseline (shift): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nBit shifts:\n%!" ;
    let time_ms, ok = run_shift_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_popcount_data () in
    Printf.printf "\nOCaml baseline (popcount): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nPopcount:\n%!" ;
    let time_ms, ok = run_popcount_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    let baseline_ms, _ = init_gray_data () in
    Printf.printf "\nOCaml baseline (gray): %.4f ms\n%!" baseline_ms ;
    Printf.printf "\nGray code:\n%!" ;
    let time_ms, ok = run_gray_code_test dev in
    Printf.printf
      "  Time: %.4f ms, Speedup: %.2fx, %s\n%!"
      time_ms
      (baseline_ms /. time_ms)
      (if ok then "PASSED" else "FAILED") ;

    print_endline "\nBitwise operations tests PASSED"
  end
