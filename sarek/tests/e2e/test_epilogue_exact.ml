(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Epilogue fusion against the unfused pair, per backend (specs/epilogue-fusion.md,
 * FR-104, AC-1). Two pairs, on inputs where fma(a, b, c) differs from
 * round(round(a * b) + c) (counted, must be positive: the positive control):
 *   forwarding pair  t = a*b ; o = t + c
 *     (the compiler could forward P's stored product into C's add);
 *   shared pair      t = a*b ; o = a*b - t
 *     (C recomputes P's product; in the fused kernel the compiler can share it,
 *      so a product with two uses is not contracted, while the unfused C may
 *      contract it into an fma residual).
 * plus a chain t = a*b ; o = t + c ; r = o * d, and an f16 pair
 *   t = f16(a*b) (float16 storage) ; o = f32(t) + c.
 *
 * On a backend where epilogue fusion is ON BY DEFAULT (Launch_epilogue.
 * default_backends: HIP, whose every kernel is compiled with
 * -ffp-contract=off) fused == unfused is ASSERTED, and the fusion must have
 * happened (1 launch instead of 2).
 * On every other deferred device the epilogue is forced on; launch counts are
 * still asserted everywhere, but bit equality is only RECORDED ("DEVIATION RECORDED n/N" when fused differs from unfused, which
 * is a precision grant the caller makes by opting in); nothing is asserted.
 * Informational: [a*b + c] in ONE kernel against the rounded and the fma result,
 * to record whether the device contracts inside a kernel.
 *
 * What it cannot detect: a deviation on inputs or shapes not listed above, a
 * backend whose compiler contracts in some other pattern, or anything on a
 * device this host does not have (CUDA runs through ZLUDA on an AMD GPU, not
 * on NVIDIA hardware; Metal is not run here).
 *   LD_LIBRARY_PATH=$HOME/opt/zluda dune exec sarek/tests/e2e/test_epilogue_exact.exe
 ******************************************************************************)

[@@@warning "-33"]

open Sarek
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector

let () =
  Test_helpers.Benchmarks.init_backends () ;
  Backend_hip.init ()

let producer =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (t : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then t.(i) <- a.(i) *. b.(i)]

let consumer =
  [%kernel
    fun (o : float32 vector)
        (t : float32 vector)
        (c : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- t.(i) +. c.(i)]

(* Float16 storage of the intermediate, narrowing by the default barrier. *)
let producer_f16 =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (t : float16 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then t.(i) <- float16_of_float32 (a.(i) *. b.(i))]

let consumer_f16 =
  [%kernel
    fun (o : float32 vector)
        (t : float16 vector)
        (c : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- float32_of_float16 t.(i) +. c.(i)]

(* The consumer recomputes the producer's product. *)
let consumer_shared =
  [%kernel
    fun (o : float32 vector)
        (a : float32 vector)
        (b : float32 vector)
        (t : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- (a.(i) *. b.(i)) -. t.(i)]

(* A second epilogue, on the consumer's output: a chain P -> C -> D. *)
let scale =
  [%kernel
    fun (r : float32 vector)
        (o : float32 vector)
        (d : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then r.(i) <- o.(i) *. d.(i)]

(* The same arithmetic in one kernel, to observe in-kernel contraction. *)
let single =
  [%kernel
    fun (a : float32 vector)
        (b : float32 vector)
        (c : float32 vector)
        (o : float32 vector)
        (n : int32) ->
      let i = global_thread_id in
      if i < n then o.(i) <- (a.(i) *. b.(i)) +. c.(i)]

let ir_of kern =
  let _, kirc = kern in
  match kirc.Kirc_types.body_ir with
  | Some ir -> ir
  | None -> failwith "kernel has no IR"

let n = 4099

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Deterministic floats in [1, 2) with full mantissas, scaled so that products
   and sums have many significant bits to lose in a contraction. *)
let next_float state =
  state := ((!state * 1103515245) + 12345) land 0x3fffffff ;
  1.0 +. (Float.of_int (!state land 0xffffff) /. 16777216.0)

let make_inputs () =
  let st = ref 12345 in
  let gen scale = Array.init n (fun _ -> round32 (scale *. next_float st)) in
  (gen 1.0, gen 1.0, gen 0.01)

let reference_rounded a b c =
  Array.init n (fun i -> round32 (round32 (a.(i) *. b.(i)) +. c.(i)))

let reference_contracted a b c =
  Array.init n (fun i -> round32 (Float.fma a.(i) b.(i) c.(i)))

let bits_of_array = Array.map Int32.bits_of_float

let to_vector arr =
  let v = Vector.create Vector.float32 (Array.length arr) in
  Array.iteri (fun i x -> Vector.set v i x) arr ;
  v

let zeros () = to_vector (Array.make n 0.0)

let bits_of_vector (v : (float, _) Vector.t) =
  Array.init (Vector.length v) (fun i -> Int32.bits_of_float (Vector.get v i))

let count_diff x y =
  let d = ref 0 in
  Array.iteri (fun i v -> if v <> y.(i) then incr d) x ;
  !d

let d1 = Execute.dims1d

let grid = (n + 255) / 256

type pair = Forwarding | Shared | Half

let launch dev ir args =
  Execute.run_vectors ~device:dev ~ir ~args ~block:(d1 256) ~grid:(d1 grid) ()

(* The float16 pair: the intermediate is stored as binary16. *)
let run_half dev (a, b, c) =
  let open Execute in
  let va = to_vector a and vb = to_vector b and vc = to_vector c in
  let vo = zeros () and vt = Vector.create Vector.float16 n in
  launch dev (ir_of producer_f16) [Vec va; Vec vb; Vec vt; Int n] ;
  launch dev (ir_of consumer_f16) [Vec vo; Vec vt; Vec vc; Int n] ;
  (bits_of_vector vt, bits_of_vector vo)

let run_pair dev pair ((a, b, c) as inputs) =
  let open Execute in
  match pair with
  | Half -> run_half dev inputs
  | Forwarding | Shared ->
      let va = to_vector a and vb = to_vector b and vc = to_vector c in
      let vt = zeros () and vo = zeros () in
      launch dev (ir_of producer) [Vec va; Vec vb; Vec vt; Int n] ;
      (match pair with
      | Shared ->
          launch
            dev
            (ir_of consumer_shared)
            [Vec vo; Vec va; Vec vb; Vec vt; Int n]
      | _ -> launch dev (ir_of consumer) [Vec vo; Vec vt; Vec vc; Int n]) ;
      (bits_of_vector vt, bits_of_vector vo)

(* t = a * b ; o = t + c ; r = o * d (three launches, two epilogues). *)
let run_chain dev (a, b, c) d =
  let open Execute in
  let va = to_vector a and vb = to_vector b and vc = to_vector c in
  let vd = to_vector d in
  let vt = zeros () and vo = zeros () and vr = zeros () in
  let launch ir args =
    run_vectors ~device:dev ~ir ~args ~block:(d1 256) ~grid:(d1 grid) ()
  in
  launch (ir_of producer) [Vec va; Vec vb; Vec vt; Int n] ;
  launch (ir_of consumer) [Vec vo; Vec vt; Vec vc; Int n] ;
  launch (ir_of scale) [Vec vr; Vec vo; Vec vd; Int n] ;
  (bits_of_vector vt, bits_of_vector vo, bits_of_vector vr)

let run_single dev (a, b, c) =
  let open Execute in
  let va = to_vector a and vb = to_vector b and vc = to_vector c in
  let vo = zeros () in
  run_vectors
    ~device:dev
    ~ir:(ir_of single)
    ~args:[Vec va; Vec vb; Vec vc; Vec vo; Int n]
    ~block:(d1 256)
    ~grid:(d1 grid)
    () ;
  bits_of_vector vo

let issued_by f =
  Execute.Window.flush () ;
  Execute.Window.reset_stats () ;
  let r = f () in
  Execute.Window.flush () ;
  (r, (Execute.Window.stats ()).issued)

let failures = ref []

let check (dev : Device.t) label ok detail =
  Printf.printf
    "  %-8s %-30s %-34s %s%s\n%!"
    dev.Device.framework
    dev.Device.name
    label
    (if ok then "PASS" else "FAIL")
    detail ;
  if not ok then failures := (dev.Device.framework ^ " " ^ label) :: !failures

(* Window and epilogue forced on for this backend, whatever the defaults. *)
let with_backend (dev : Device.t) ~epilogue f =
  let backend = dev.Device.framework in
  let window_was = Execute.Window.defers backend
  and epilogue_was = Execute.Window.epilogue_on backend in
  Execute.Window.set_backend_enabled ~backend true ;
  Execute.Window.set_epilogue_backend ~backend epilogue ;
  Fun.protect
    ~finally:(fun () ->
      Execute.Window.set_epilogue_backend ~backend epilogue_was ;
      Execute.Window.set_backend_enabled ~backend window_was)
    f

(* Asserted on a backend where the epilogue is on by default, recorded
   elsewhere. *)
let gated (dev : Device.t) =
  List.mem dev.Device.framework Sarek_execute.Launch_epilogue.default_backends

let verdict (dev : Device.t) label ok detail =
  if gated dev then check dev label ok detail
  else
    Printf.printf
      "  %-8s %-30s %-34s %s%s\n%!"
      dev.Device.framework
      dev.Device.name
      label
      (if ok then "match" else "DEVIATION RECORDED")
      detail

let run_pair_device (dev : Device.t) pair label inputs =
  let (t_off, o_off), launches_off =
    with_backend dev ~epilogue:false (fun () ->
        issued_by (fun () -> run_pair dev pair inputs))
  in
  let (t_on, o_on), launches_on =
    with_backend dev ~epilogue:true (fun () ->
        issued_by (fun () -> run_pair dev pair inputs))
  in
  check dev (label ^ ": off = 2 launches") (launches_off = 2) "" ;
  check
    dev
    (label ^ ": on = 1 launch")
    (launches_on = 1)
    (Printf.sprintf " (%d)" launches_on) ;
  verdict
    dev
    (label ^ ": fused == unfused")
    (t_on = t_off && o_on = o_off)
    (Printf.sprintf
       " (%d of %d elements of o differ)"
       (count_diff o_on o_off)
       n) ;
  o_off

let chain_reference (a, b, c) d =
  Array.init n (fun i ->
      round32 (round32 (round32 (a.(i) *. b.(i)) +. c.(i)) *. d.(i)))

let run_chain_device (dev : Device.t) inputs =
  let d =
    Array.init n (fun i -> round32 (1.0 +. (Float.of_int (i mod 7) /. 64.0)))
  in
  let off, launches_off =
    with_backend dev ~epilogue:false (fun () ->
        issued_by (fun () -> run_chain dev inputs d))
  in
  let on, launches_on =
    with_backend dev ~epilogue:true (fun () ->
        issued_by (fun () -> run_chain dev inputs d))
  in
  let _, _, r_off = off in
  check dev "chain: off = 3 launches" (launches_off = 3) "" ;
  check
    dev
    "chain: on = 1 launch"
    (launches_on = 1)
    (Printf.sprintf " (%d)" launches_on) ;
  verdict dev "chain: fused == unfused" (on = off) "" ;
  verdict
    dev
    "chain: unfused == rounded reference"
    (r_off = bits_of_array (chain_reference inputs d))
    ""

(* Informational: does the device contract a*b+c inside one kernel? *)
let observe_contraction (dev : Device.t) inputs ~rounded ~contracted =
  let o = run_single dev inputs in
  Printf.printf
    "  %-8s %-30s in-kernel a*b+c vs rounded: %d differ; vs fma: %d differ \
     (INFO)\n\
     %!"
    dev.Device.framework
    dev.Device.name
    (count_diff o (bits_of_array rounded))
    (count_diff o (bits_of_array contracted))

let run_all_pairs (dev : Device.t) inputs ~rounded =
  let o_fwd = run_pair_device dev Forwarding "forwarding" inputs in
  verdict
    dev
    "forwarding: unfused == rounded reference"
    (o_fwd = bits_of_array rounded)
    "" ;
  ignore (run_pair_device dev Shared "shared" inputs) ;
  run_chain_device dev inputs ;
  (* Last. OpenCL and Vulkan refuse float16 kernels by policy: a skip there,
     a failure on a default backend. *)
  match run_pair_device dev Half "f16" inputs with
  | _ -> ()
  | exception e when not (gated dev) ->
      Printf.printf
        "  %-8s f16: SKIP (%s)\n%!"
        dev.framework
        (let m = Printexc.to_string e in
         String.sub m 0 (min 60 (String.length m)))

(* A failure of a backend that cannot run the scenario is a skip; anything else
   is a failure. *)
let guarded (dev : Device.t) f =
  match f () with
  | () -> ()
  | exception e ->
      let regression = Test_helpers.Backend_failure.verdict e = `Fail in
      Printf.printf
        "  %-8s %s: %s\n%!"
        dev.framework
        (if regression then "FAIL" else "SKIP")
        (Printexc.to_string e) ;
      if regression then failures := (dev.framework ^ " raised") :: !failures

let on_device inputs ~rounded ~contracted (dev : Device.t) =
  if Device.is_cpu dev || dev.framework = "Interpreter" then
    Printf.printf
      "  %-8s %-30s SKIP (launches are not deferred)\n%!"
      dev.framework
      dev.name
  else begin
    guarded dev (fun () -> run_all_pairs dev inputs ~rounded) ;
    guarded dev (fun () -> observe_contraction dev inputs ~rounded ~contracted)
  end

let report_and_exit () =
  if !failures <> [] then begin
    Printf.printf
      "test_epilogue_exact: FAILED (%s)\n"
      (String.concat ", " (List.rev !failures)) ;
    exit 1
  end ;
  print_endline "test_epilogue_exact: PASSED"

let () =
  print_endline "=== epilogue fusion exactness, E2E ===" ;
  let ((a, b, c) as inputs) = make_inputs () in
  let rounded = reference_rounded a b c
  and contracted = reference_contracted a b c in
  let sensitive =
    count_diff (bits_of_array rounded) (bits_of_array contracted)
  in
  Printf.printf
    "  inputs: %d of %d elements differ between fma and mul-then-add\n%!"
    sensitive
    n ;
  if sensitive = 0 then
    failures := "no contraction-sensitive input" :: !failures ;
  Array.iter (on_device inputs ~rounded ~contracted) (Device.init ()) ;
  report_and_exit ()
