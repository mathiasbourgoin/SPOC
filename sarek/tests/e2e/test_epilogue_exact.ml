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
 * On a backend where epilogue fusion is exact by default (Sarek_policy.requires
 * Epilogue_fusion = Some []: HIP, whose every kernel is compiled with
 * -ffp-contract=off, and Metal, whose every kernel carries
 * #pragma METAL fp contract(off)) fused == unfused AND fused == Interpreter
 * (which rounds every operation as written) are ASSERTED bit for bit, and the
 * fusion must have happened (1 launch instead of 2). Positive controls: the
 * inputs separate fma from mul-then-add, and the Interpreter and the fused
 * forwarding result both differ from the fma reference. The fp-contract pragma in the fused Metal sources is checked without a
 * device by test_epilogue_metal_pragma.
 * On every other deferred device the epilogue is forced on after the program
 * grants Cross_kernel_contraction (without it nothing fuses there); launch
 * counts are asserted, bit equality only RECORDED ("DEVIATION RECORDED"), and
 * the fused result is asserted to lie within one contraction per element of
 * the unfused one (|f - u| <= ulp(a*b)/2 + ulp(result): not 1 ulp of the
 * result, which cancellation breaks; the number of differing elements and the
 * largest distance in ulps are printed). Informational: [a*b + c] in ONE kernel
 * against the rounded and the fma result (does the device contract?).
 *
 * What it cannot detect: a deviation on inputs or shapes not listed above, a
 * backend whose compiler contracts in some other pattern, or anything on a
 * device this host does not have (CUDA runs through ZLUDA on an AMD GPU, not
 * on NVIDIA hardware; Metal runs only on a Mac, by hand).
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

(* The same launch on the Interpreter, which rounds every operation as written:
   the reference a default backend's fused result must equal. *)
let launch_interp ir args =
  Execute.run_interpreter_vectors
    ~ir
    ~args
    ~block:(d1 256)
    ~grid:(d1 grid)
    ~parallel:false

(* The float16 pair: the intermediate is stored as binary16. *)
let run_half launch (a, b, c) =
  let open Execute in
  let va = to_vector a and vb = to_vector b and vc = to_vector c in
  let vo = zeros () and vt = Vector.create Vector.float16 n in
  launch (ir_of producer_f16) [Vec va; Vec vb; Vec vt; Int n] ;
  launch (ir_of consumer_f16) [Vec vo; Vec vt; Vec vc; Int n] ;
  (bits_of_vector vt, bits_of_vector vo)

let run_pair launch pair ((a, b, c) as inputs) =
  let open Execute in
  match pair with
  | Half -> run_half launch inputs
  | Forwarding | Shared ->
      let va = to_vector a and vb = to_vector b and vc = to_vector c in
      let vt = zeros () and vo = zeros () in
      launch (ir_of producer) [Vec va; Vec vb; Vec vt; Int n] ;
      (match pair with
      | Shared ->
          launch (ir_of consumer_shared) [Vec vo; Vec va; Vec vb; Vec vt; Int n]
      | _ -> launch (ir_of consumer) [Vec vo; Vec vt; Vec vc; Int n]) ;
      (bits_of_vector vt, bits_of_vector vo)

(* t = a * b ; o = t + c ; r = o * d (three launches, two epilogues). *)
let run_chain launch (a, b, c) d =
  let open Execute in
  let va = to_vector a and vb = to_vector b and vc = to_vector c in
  let vd = to_vector d in
  let vt = zeros () and vo = zeros () and vr = zeros () in
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

(* The backends whose kernels are never contracted fuse without any grant. *)
let exact_by_default (dev : Device.t) =
  Sarek_policy.requires
    Sarek_policy.Epilogue_fusion
    ~backend:dev.Device.framework
  = Some []

(* Elsewhere fusion needs the program's grant that one multiply of a kernel may
   be contracted with an add of the next (a deviation bounded by one
   contraction per element, see [check_bound]). It is made once, for the
   process, when the epilogue is first forced on for such a backend. *)
let grant_contraction (dev : Device.t) =
  if not (exact_by_default dev) then
    ignore
      (Sarek_policy.grant
         Sarek_policy.Process
         (Sarek_policy.Deviation Sarek_policy.Cross_kernel_contraction)
         ~reason:"test_epilogue_exact: measuring the contraction bound")

(* Window and epilogue forced on for this backend, whatever the defaults. *)
let with_backend (dev : Device.t) ~epilogue f =
  let backend = dev.Device.framework in
  if epilogue then grant_contraction dev ;
  let window_was = Execute.Window.defers backend
  and epilogue_was = Execute.Window.epilogue_on backend in
  Execute.Window.set_backend_enabled ~backend true ;
  Execute.Window.set_epilogue_backend ~backend epilogue ;
  Fun.protect
    ~finally:(fun () ->
      Execute.Window.set_epilogue_backend ~backend epilogue_was ;
      Execute.Window.set_backend_enabled ~backend window_was)
    f

(* Bit equality is asserted on a backend where the epilogue is exact by
   default, recorded elsewhere (where [check_bound] is asserted instead). *)
let gated = exact_by_default

(* Position of a float32 on a line that is monotone in the float order, so that
   the distance of two positions is their distance in ulps. *)
let ordered bits =
  let i = Int32.to_int bits in
  if i >= 0 then i else -0x80000000 - i

let ulp_distance x y = abs (ordered x - ordered y)

(* "One contraction per element". With u the unfused result round32(round32(a*b)
   + c) and f the fused one round32(a*b + c): |f - u| <= ulp(a*b)/2 + ulp(r),
   r the larger of |f| and |u| (test_contraction_bound proves it on the host).
   It is NOT 1 ulp of the result: under cancellation the difference is a rounding
   error of the product. For a result r = o * d computed from such an o, the
   error of o is scaled by |d|. [Ulps k] is for pairs with no product left to
   contract (the float16 pair). *)
type bound =
  | Ulps of int
  | Contraction of float array * float array
  | Chained of float array * float array * float array * int32 array

let ulp x =
  if x = 0.0 then Float.ldexp 1.0 (-149)
  else
    let _, e = Float.frexp (Float.abs x) in
    Float.ldexp 1.0 (max (e - 24) (-149))

let value = Int32.float_of_bits

(* The largest |difference| relative to what the bound allows (<= 1 holds). *)
let measure bound fused unfused =
  let worst = ref 0.0 in
  Array.iteri
    (fun i v ->
      let u = unfused.(i) in
      let delta = Float.abs (value v -. value u) in
      let r = Float.max (Float.abs (value v)) (Float.abs (value u)) in
      let allowed =
        match bound with
        | Ulps k -> float_of_int k *. ulp r
        | Contraction (a, b) -> (ulp (a.(i) *. b.(i)) /. 2.0) +. ulp r
        | Chained (a, b, d, o) ->
            Float.abs d.(i)
            *. ((ulp (a.(i) *. b.(i)) /. 2.0) +. ulp (value o.(i)))
            +. ulp r
      in
      worst := Float.max !worst (delta /. allowed))
    fused ;
  !worst

(* Asserted on every backend that fuses: the bound holds, and what was measured
   is printed (differing elements, largest difference in ulps of the result,
   and relative to the bound). *)
let check_bound dev label bound ~fused ~unfused =
  let worst = measure bound fused unfused in
  check
    dev
    (label ^ ": within one contraction")
    (worst <= 1.0)
    (Printf.sprintf
       " (%d of %d differ, max %d ulp, %.3g of the bound)"
       (count_diff fused unfused)
       n
       (Array.fold_left
          max
          0
          (Array.mapi (fun i v -> ulp_distance v unfused.(i)) fused))
       worst)

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

(* What the Interpreter computes for each scenario (every operation rounded as
   written), the reference a default backend's fused result must equal, and
   the fma-contracted forwarding result, which no exact path may equal. *)
type references = {
  forwarding : int32 array;
  shared : int32 array;
  half : int32 array option;
  chain : int32 array;
  contracted : int32 array;
}

let reference_of refs = function
  | Forwarding -> Some refs.forwarding
  | Shared -> Some refs.shared
  | Half -> refs.half

let run_pair_device (dev : Device.t) refs pair label inputs =
  let (t_off, o_off), launches_off =
    with_backend dev ~epilogue:false (fun () ->
        issued_by (fun () -> run_pair (launch dev) pair inputs))
  in
  let (t_on, o_on), launches_on =
    with_backend dev ~epilogue:true (fun () ->
        issued_by (fun () -> run_pair (launch dev) pair inputs))
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
  let a, b, _ = inputs in
  check_bound
    dev
    label
    (match pair with
    | Half -> Ulps 1
    | Forwarding | Shared -> Contraction (a, b))
    ~fused:o_on
    ~unfused:o_off ;
  Option.iter
    (fun reference ->
      verdict
        dev
        (label ^ ": fused == Interpreter")
        (o_on = reference)
        (Printf.sprintf
           " (%d of %d elements differ)"
           (count_diff o_on reference)
           n) ;
      verdict dev (label ^ ": unfused == Interpreter") (o_off = reference) "")
    (reference_of refs pair) ;
  (o_off, o_on)

let chain_reference (a, b, c) d =
  Array.init n (fun i ->
      round32 (round32 (round32 (a.(i) *. b.(i)) +. c.(i)) *. d.(i)))

let chain_scale () =
  Array.init n (fun i -> round32 (1.0 +. (Float.of_int (i mod 7) /. 64.0)))

let run_chain_device (dev : Device.t) refs inputs =
  let d = chain_scale () in
  let off, launches_off =
    with_backend dev ~epilogue:false (fun () ->
        issued_by (fun () -> run_chain (launch dev) inputs d))
  in
  let on, launches_on =
    with_backend dev ~epilogue:true (fun () ->
        issued_by (fun () -> run_chain (launch dev) inputs d))
  in
  let _, _, r_off = off and _, _, r_on = on in
  check dev "chain: off = 3 launches" (launches_off = 3) "" ;
  check
    dev
    "chain: on = 1 launch"
    (launches_on = 1)
    (Printf.sprintf " (%d)" launches_on) ;
  verdict dev "chain: fused == unfused" (on = off) "" ;
  (* Two epilogues: the contraction of the first reaches the second's input. *)
  let _, o_off, _ = off and _, o_on, _ = on in
  let a, b, _ = inputs in
  check_bound dev "chain o" (Contraction (a, b)) ~fused:o_on ~unfused:o_off ;
  check_bound dev "chain r" (Chained (a, b, d, o_on)) ~fused:r_on ~unfused:r_off ;
  verdict dev "chain: fused == Interpreter" (r_on = refs.chain) "" ;
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

(* OpenCL and Vulkan refuse float16 kernels by policy: a skip there, a failure
   on a default backend (HIP, Metal), where the pair is asserted. *)
let run_half_pair dev refs inputs =
  match run_pair_device dev refs Half "f16" inputs with
  | _ -> ()
  | exception e when not (gated dev) ->
      Printf.printf
        "  %-8s f16: SKIP (%s)\n%!"
        dev.framework
        (let m = Printexc.to_string e in
         String.sub m 0 (min 60 (String.length m)))

(* Positive control: the comparison must be able to go red. On a default
   backend the fused forwarding result must DIFFER from the fma-contracted
   reference (on the inputs where fma and mul-then-add disagree); a device that
   contracted would equal it and fail both this and the Interpreter check. *)
let control_not_contracted dev refs o_fused =
  let differ = count_diff o_fused refs.contracted in
  verdict
    dev
    "control: fused != fma reference"
    (differ > 0)
    (Printf.sprintf " (%d of %d differ)" differ n)

let run_all_pairs (dev : Device.t) refs inputs ~rounded =
  let o_fwd, o_fwd_fused =
    run_pair_device dev refs Forwarding "forwarding" inputs
  in
  verdict
    dev
    "forwarding: unfused == rounded reference"
    (o_fwd = bits_of_array rounded)
    "" ;
  control_not_contracted dev refs o_fwd_fused ;
  ignore (run_pair_device dev refs Shared "shared" inputs) ;
  run_chain_device dev refs inputs ;
  (* Last: a refused scenario must not hide the others. *)
  run_half_pair dev refs inputs

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

let on_device refs inputs ~rounded ~contracted (dev : Device.t) =
  if Device.is_cpu dev || dev.framework = "Interpreter" then
    Printf.printf
      "  %-8s %-30s SKIP (launches are not deferred)\n%!"
      dev.framework
      dev.name
  else begin
    guarded dev (fun () -> run_all_pairs dev refs inputs ~rounded) ;
    guarded dev (fun () -> observe_contraction dev inputs ~rounded ~contracted)
  end

let interpreter_references inputs ~contracted =
  let run pair = snd (run_pair launch_interp pair inputs) in
  let half =
    match run Half with
    | bits -> Some bits
    | exception e ->
        Printf.printf
          "  Interpreter f16 reference: SKIP (%s)\n%!"
          (Printexc.to_string e) ;
        None
  in
  let _, _, chain = run_chain launch_interp inputs (chain_scale ()) in
  {
    forwarding = run Forwarding;
    shared = run Shared;
    half;
    chain;
    contracted = bits_of_array contracted;
  }

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
  let refs = interpreter_references inputs ~contracted in
  (* The Interpreter rounds as written: it equals the rounded reference and
     differs from the fma one (the control that the Interpreter reference is
     itself not contracted). *)
  let interp_ok = refs.forwarding = bits_of_array rounded in
  let interp_control = count_diff refs.forwarding refs.contracted > 0 in
  if not interp_ok then
    failures := "Interpreter != rounded reference" :: !failures ;
  if not interp_control then
    failures := "Interpreter == fma reference" :: !failures ;
  Array.iter (on_device refs inputs ~rounded ~contracted) (Device.init ()) ;
  report_and_exit ()
