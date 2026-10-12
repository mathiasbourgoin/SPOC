(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile time of three decode-shaped kernels on Metal, cold vs warm, each in
 * a NEW process.
 *
 * Question: would an on-disk compile cache of ours (a metallib or a
 * MTLBinaryArchive through Framework_cache) save anything the macOS shader
 * cache does not already save? Metal builds MSL with newLibraryWithSource and
 * then a pipeline state; macOS keeps its own cache of both, keyed by the
 * source. A cache of ours only pays if the second process is still slow.
 *
 * The parent spawns this executable as a child three times per round with the
 * same nonce, a volatile store of the nonce injected at the top of every
 * kernel body so that a nonce the machine has never seen cannot hit the macOS
 * cache (the compiled code differs, not only the text). A throwaway kernel is
 * compiled first and the kernel order rotates each round, so a once-per-process
 * cost is not attributed to one kernel. Times use CLOCK_UPTIME_RAW:
 *   cold   first process with this nonce (sources new to the OS cache)
 *   warm1  second process, same nonce
 *   warm2  third process, same nonce
 * Each child reports, per kernel, the time of [newLibraryWithSource] (MSL ->
 * library) and of the pipeline-state creation (library -> GPU code) in ms. The
 * parent prints the median and min-max over the rounds of each (phase, kernel,
 * step).
 *
 * Information only. Self-skips without a Metal device. Run by hand on a Mac:
 *   dune build @sarek-metal/test/e2e-metal-bench --force
 * SPOC_BENCH_ROUNDS (default 5) sets the number of rounds. Record `uptime`
 * next to the output.
 ******************************************************************************)

module Device = Spoc_core.Device
module Api = Sarek_metal.Metal_api

let () = Sarek_metal.Metal_plugin.init ()

(* Q8_0-shaped matvec row: int8 weights unpacked from int32 words, a scale per
   block of 32, accumulated in float. *)
let matvec_q8 =
  [%kernel
    fun (out : float32 vector)
        (wq : int32 vector)
        (scales : float32 vector)
        (x : float32 vector)
        (cols : int32)
        (rows : int32) ->
      let row = global_thread_id in
      if row < rows then begin
        let acc = mut 0.0 in
        let blocks = cols / 32l in
        let b = mut 0l in
        while b < blocks do
          let s = scales.((row * blocks) + b) in
          let part = mut 0.0 in
          let j = mut 0l in
          while j < 32l do
            let idx = (row * cols) + (b * 32l) + j in
            let q = wq.(idx / 4l) in
            let byte = (q lsr (8l * (idx land 3l))) land 255l in
            let sval = if byte > 127l then byte - 256l else byte in
            part := part +. (float sval *. x.((b * 32l) + j)) ;
            j := j + 1l
          done ;
          acc := acc +. (s *. part) ;
          b := b + 1l
        done ;
        out.(row) <- acc
      end]

(* RMS norm, one row per thread: two passes and a sqrt. *)
let rms_norm =
  [%kernel
    fun (out : float32 vector)
        (inp : float32 vector)
        (w : float32 vector)
        (n : int32)
        (rows : int32)
        (eps : float32) ->
      let row = global_thread_id in
      if row < rows then begin
        let ss = mut 0.0 in
        let i = mut 0l in
        while i < n do
          let v = inp.((row * n) + i) in
          ss := ss +. (v *. v) ;
          i := i + 1l
        done ;
        let scale = 1.0 /. sqrt ((ss /. float n) +. eps) in
        let k = mut 0l in
        while k < n do
          out.((row * n) + k) <- inp.((row * n) + k) *. scale *. w.(k) ;
          k := k + 1l
        done
      end]

(* RoPE-shaped rotation: sin/cos per pair. *)
let rope =
  [%kernel
    fun (q : float32 vector)
        (pos : int32)
        (half_dim : int32)
        (theta : float32)
        (n : int32) ->
      let tid = global_thread_id in
      if tid < n then begin
        let pair = tid mod half_dim in
        let freq = pow theta (0.0 -. (float pair /. float half_dim)) in
        let angle = float pos *. freq in
        let c = cos angle in
        let s = sin angle in
        let a = q.(2l * tid) in
        let b = q.((2l * tid) + 1l) in
        q.(2l * tid) <- (a *. c) -. (b *. s) ;
        q.((2l * tid) + 1l) <- (a *. s) +. (b *. c)
      end]

let kernels = [("matvec_q8", matvec_q8); ("rms_norm", rms_norm); ("rope", rope)]

let ir_of (_, kirc) =
  match kirc.Sarek.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "no IR"

(* Monotonic clock (CLOCK_UPTIME_RAW), not the wall clock. *)
let ms f =
  let t0 = Sarek_metal.Metal_bindings.uptime_raw_seconds () in
  let r = f () in
  (r, (Sarek_metal.Metal_bindings.uptime_raw_seconds () -. t0) *. 1000.)

(* The entry point is the first [kernel void NAME(] of the source. *)
let entry_name source =
  let marker = "kernel void " in
  let start = Str.search_forward (Str.regexp_string marker) source 0 in
  let from = start + String.length marker in
  String.sub source from (String.index_from source from '(' - from)

(* A nonce that changes the compiled code, not just the text: a volatile store
   of the nonce at the top of the kernel body. A comment would change the source
   hash but leave the compiled library, and so a pipeline cache keyed on it,
   identical. *)
let with_nonce source ~nonce =
  let entry_start =
    Str.search_forward (Str.regexp_string "kernel void ") source 0
  in
  let body = String.index_from source entry_start '{' + 1 in
  String.sub source 0 body
  ^ Printf.sprintf " volatile uint nonce_k = %du; (void)nonce_k; " nonce
  ^ String.sub source body (String.length source - body)

(* A throwaway kernel timed by nobody: it pays the once-per-process cost of
   the first Metal compile, so it is not attributed to the first kernel. *)
let warm_up_source ~nonce =
  with_nonce
    "kernel void warm_up(device float* o [[buffer(0)]], uint t \
     [[thread_position_in_grid]]) { o[t] = 1.0f; }"
    ~nonce

let rotate k xs =
  let n = List.length xs in
  List.init n (fun i -> List.nth xs ((i + k) mod n))

let compile_timed mdev source =
  let library, lib_ms =
    ms (fun () -> Api.Library.create_from_source mdev source)
  in
  let func = Api.Library.get_function library (entry_name source) in
  let _, pso_ms = ms (fun () -> Api.ComputePipeline.create mdev func) in
  (lib_ms, pso_ms)

(* ---- child: one process, one nonce, kernels in a rotated order ---- *)

let child ~nonce ~rotation =
  let devices = Device.init () in
  match
    Array.to_list devices
    |> List.find_opt (fun d -> d.Device.framework = "Metal")
  with
  | None -> print_endline "SKIP"
  | Some dev ->
      let mdev = Api.Device.get dev.Device.backend_id in
      ignore (compile_timed mdev (warm_up_source ~nonce)) ;
      List.iter
        (fun (label, k) ->
          let ir = ir_of k in
          let source =
            with_nonce
              (Sarek_metal.Sarek_ir_metal.generate_with_types
                 ~types:ir.kern_types
                 ir)
              ~nonce
          in
          let lib_ms, pso_ms = compile_timed mdev source in
          Printf.printf "RESULT %s library %.3f\n" label lib_ms ;
          Printf.printf "RESULT %s pipeline %.3f\n%!" label pso_ms)
        (rotate rotation kernels)

(* ---- parent ---- *)

let run_child ~nonce ~rotation =
  let ic =
    Unix.open_process_args_in
      Sys.executable_name
      [|
        Sys.executable_name;
        "--child";
        string_of_int nonce;
        string_of_int rotation;
      |]
  in
  let rec lines acc =
    match input_line ic with
    | l -> lines (l :: acc)
    | exception End_of_file -> List.rev acc
  in
  let out = lines [] in
  (match Unix.close_process_in ic with
  | Unix.WEXITED 0 -> ()
  | _ ->
      failwith
        (Printf.sprintf
           "bench_metal_compile: child (nonce %d) failed; its output:\n%s"
           nonce
           (String.concat "\n" out))) ;
  out

let parse_result l =
  match String.split_on_char ' ' l with
  | ["RESULT"; kernel; step; t] -> Some ((kernel, step), float_of_string t)
  | _ -> None

let median xs =
  let a = Array.of_list (List.sort compare xs) in
  a.(Array.length a / 2)

let rounds =
  match Sys.getenv_opt "SPOC_BENCH_ROUNDS" with
  | Some s -> int_of_string s
  | None -> 5

let phases = ["cold"; "warm1"; "warm2"]

let record samples phase out =
  List.iter
    (fun l ->
      match parse_result l with
      | Some (key, t) ->
          let k = (phase, key) in
          Hashtbl.replace
            samples
            k
            (t :: Option.value ~default:[] (Hashtbl.find_opt samples k))
      | None -> ())
    out

let print_row samples (label, step) =
  Printf.printf "  %-10s %-8s" label step ;
  List.iter
    (fun phase ->
      let xs = Hashtbl.find samples (phase, (label, step)) in
      Printf.printf
        "  %s %.2f [%.2f-%.2f]"
        phase
        (median xs)
        (List.fold_left Float.min infinity xs)
        (List.fold_left Float.max neg_infinity xs))
    phases ;
  print_newline ()

let parent () =
  let samples = Hashtbl.create 64 in
  let skipped = ref false in
  for round = 1 to rounds do
    let nonce = (int_of_float (Unix.gettimeofday ()) + round) land 0x3fffffff in
    let rotation = round - 1 in
    List.iter
      (fun phase ->
        let out = run_child ~nonce ~rotation in
        if List.mem "SKIP" out then skipped := true ;
        record samples phase out)
      phases
  done ;
  if !skipped then print_endline "bench_metal_compile: [SKIP] no Metal device"
  else begin
    Printf.printf
      "Metal compile time per kernel, new process each, %d rounds (ms, median \
       [min-max])\n"
      rounds ;
    List.iter
      (fun (label, _) ->
        List.iter
          (fun step -> print_row samples (label, step))
          ["library"; "pipeline"])
      kernels
  end

let () =
  match Array.to_list Sys.argv with
  | [_; "--child"; nonce; rotation] ->
      child ~nonce:(int_of_string nonce) ~rotation:(int_of_string rotation)
  | _ -> parent ()
