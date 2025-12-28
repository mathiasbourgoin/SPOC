(******************************************************************************
 * Test helpers for Sarek E2E tests
 *
 * Shared utilities for device selection, verification, and benchmarking.
 ******************************************************************************)

open Spoc

(** Command line options *)
type config = {
  mutable dev_id : int;
  mutable use_interpreter : bool;
  mutable use_native : bool;
  mutable use_native_parallel : bool;
  mutable benchmark_all : bool;
  mutable verify : bool;
  mutable size : int;
  mutable block_size : int;
}

let default_config () =
  {
    dev_id = 0;
    use_interpreter = false;
    use_native = false;
    use_native_parallel = false;
    benchmark_all = false;
    verify = true;
    size = 1024;
    block_size = 256;
  }

let usage name extra_opts =
  Printf.printf "Usage: %s [options]\n" name ;
  Printf.printf "Options:\n" ;
  Printf.printf "  -d <id>       Device ID (default: 0)\n" ;
  Printf.printf "  --interpreter Use CPU interpreter device\n" ;
  Printf.printf "  --native      Use native CPU runtime device (sequential)\n" ;
  Printf.printf
    "  --native-parallel  Use native CPU runtime with parallel threads\n" ;
  Printf.printf "  --benchmark   Run on all devices and compare times\n" ;
  Printf.printf "  -s <size>     Problem size (default: 1024)\n" ;
  Printf.printf "  -b <size>     Block/work-group size (default: 256)\n" ;
  Printf.printf "  -no-verify    Skip result verification\n" ;
  Printf.printf "  -h            Show this help\n" ;
  extra_opts () ;
  exit 0

let parse_args ?(extra = fun _ _ -> false) ?(extra_usage = fun () -> ()) name =
  let cfg = default_config () in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    let consumed = extra cfg !i in
    (if not consumed then
       match Sys.argv.(!i) with
       | "-d" ->
           incr i ;
           cfg.dev_id <- int_of_string Sys.argv.(!i)
       | "--interpreter" -> cfg.use_interpreter <- true
       | "--native" -> cfg.use_native <- true
       | "--native-parallel" -> cfg.use_native_parallel <- true
       | "--benchmark" -> cfg.benchmark_all <- true
       | "-s" ->
           incr i ;
           cfg.size <- int_of_string Sys.argv.(!i)
       | "-b" ->
           incr i ;
           cfg.block_size <- int_of_string Sys.argv.(!i)
       | "-no-verify" -> cfg.verify <- false
       | "-h" | "--help" -> usage name extra_usage
       | _ -> ()) ;
    incr i
  done ;
  cfg

(** Get device based on config *)
let get_device cfg devs =
  if cfg.use_native_parallel then Devices.create_native_device ~parallel:true ()
  else if cfg.use_native then (
    match Devices.find_native_id devs with
    | Some id -> devs.(id)
    | None ->
        print_endline "No native CPU device found" ;
        exit 1)
  else if cfg.use_interpreter then (
    match Devices.find_interpreter_id devs with
    | Some id -> devs.(id)
    | None ->
        print_endline "No interpreter device found" ;
        exit 1)
  else devs.(cfg.dev_id)

(** Print available devices *)
let print_devices devs =
  Printf.printf "Available devices:\n" ;
  Array.iteri
    (fun i d ->
      Printf.printf "  [%d] %s\n" i d.Devices.general_info.Devices.name)
    devs ;
  flush stdout

(** Run benchmark on all devices *)
let benchmark_all devs run_test name =
  Printf.printf "\nBenchmark: %s\n" name ;
  Printf.printf "%-40s %12s %10s\n" "Device" "Time (ms)" "Status" ;
  Printf.printf "%s\n" (String.make 64 '-') ;
  (* Run on all devices (including parallel native from Devices.init) *)
  Array.iter
    (fun dev ->
      let dev_name = dev.Devices.general_info.Devices.name in
      flush stdout ;
      let time_ms, ok = run_test dev in
      let status = if ok then "OK" else "FAIL" in
      Printf.printf "%-40s %12.4f %10s\n%!" dev_name time_ms status)
    devs ;
  print_endline "\nBenchmark complete."

(** Get appropriate block size for device *)
let get_block_size cfg dev =
  match dev.Devices.specific_info with
  | Devices.OpenCLInfo clI -> (
      match clI.Devices.device_type with
      | Devices.CL_DEVICE_TYPE_CPU ->
          (* CPU OpenCL can use larger work-groups for barrier-based kernels.
             Use cfg.block_size if specified, otherwise default to reasonable size. *)
          if cfg.block_size > 1 then cfg.block_size else 64
      | _ -> cfg.block_size)
  | _ -> cfg.block_size

(** Verify float vectors are approximately equal *)
let verify_float_vector expected actual tolerance =
  let n = Vector.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = Mem.get expected i in
    let a = Mem.get actual i in
    if abs_float (e -. a) > tolerance then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %.6f, got %.6f\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Verify int32 vectors are equal *)
let verify_int32_vector expected actual =
  let n = Vector.length expected in
  let errors = ref 0 in
  for i = 0 to n - 1 do
    let e = Mem.get expected i in
    let a = Mem.get actual i in
    if e <> a then begin
      if !errors < 10 then
        Printf.printf "  Mismatch at %d: expected %ld, got %ld\n" i e a ;
      incr errors
    end
  done ;
  if !errors > 0 then Printf.printf "  Total errors: %d\n" !errors ;
  !errors = 0

(** Time a function and return (result, time_ms) *)
let time_it f =
  let t0 = Unix.gettimeofday () in
  let result = f () in
  let t1 = Unix.gettimeofday () in
  (result, (t1 -. t0) *. 1000.0)
