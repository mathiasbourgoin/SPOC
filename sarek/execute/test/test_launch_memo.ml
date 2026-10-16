(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Launch-preparation memo in Execute.run (specs/cheap-repeated-launches.md).
 *
 * A JIT probe backend counts [generate_source] calls and records the source
 * string each launch receives, so every assertion below is about what Execute
 * actually did, not about what it returned. Kernels take scalar arguments
 * only: vector arguments would need real device buffers.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework.Framework_sig
open Spoc_framework_registry
module Execute = Sarek_execute.Execute
module Device = Spoc_core.Device

let backend_name = "LaunchMemoProbe"

let codegen_calls = Atomic.make 0

(* Runs inside generate_source, before it returns (FR-011 race test). *)
let codegen_hook : (unit -> unit) ref = ref (fun () -> ())

let launched_lock = Mutex.create ()

let codegen_mode : [`Ok | `None | `Raise | `Cuda] ref = ref `Ok

(* Source strings received by run_source, most recent first. *)
let launched : string list ref = ref []

module Probe : BACKEND = struct
  let name = backend_name

  let version = (1, 0, 0)

  let is_available () = true

  let execution_model = JIT

  let supported_source_langs = [CUDA_Source]

  let generate_source ?block ?soa_params:_ (ir : Sarek_ir_types.kernel) =
    Atomic.incr codegen_calls ;
    !codegen_hook () ;
    match !codegen_mode with
    | `None -> None
    | `Raise -> failwith "probe codegen refused"
    | `Cuda ->
        Some (Sarek_codegen.Sarek_ir_cuda.generate_with_types ~types:[] ir)
    | `Ok ->
        let bx = match block with Some b -> b.x | None -> 0 in
        (* A FRESH string per call, so physical equality across launches can
           only come from the memo. *)
        Some (Printf.sprintf "// %s block=%d\n" ir.Sarek_ir_types.kern_name bx)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args = ()

  let wrap_kargs () = failwith "probe: wrap_kargs"

  let unwrap_kargs _ = None

  let run_source ~source ~lang:_ ~kernel_name:_ ~block:_ ~grid:_ ~shared_mem:_ _
      =
    Mutex.protect launched_lock (fun () -> launched := source :: !launched)

  module Intrinsics = Intrinsic_registry.Make ()

  module Device = struct
    type t = unit

    type id = int

    let init () = ()

    let count () = 1

    let get _id = ()

    let id () = 0

    let set_current () = ()

    let synchronize () = ()

    let name () = "Launch memo probe"

    let capabilities () =
      {
        max_threads_per_block = 256;
        max_block_dims = (256, 256, 64);
        max_grid_dims = (65535, 65535, 65535);
        shared_mem_per_block = 16384;
        total_global_mem = 1073741824L;
        (* 1GB *)
        compute_capability = (0, 0);
        device_features = [];
        (* backlog-62: no cooperative-matrix probe on this backend. [None] is
           "not probed", which Sarek_coopmat.verdict maps to Unknown and therefore
           refuses; an empty list would be a positive claim nobody measured. *)
        coopmat = None;
        supports_atomics = false;
        warp_size = 1;
        max_registers_per_block = 0;
        clock_rate_khz = 0;
        multiprocessor_count = 1;
        is_cpu = true;
      }
  end

  module Stream = struct
    type t = unit

    let create () = ()

    let create_non_blocking () = ()

    let default () = ()

    let synchronize () = ()

    let destroy () = ()
  end

  module Memory = struct
    type 'a buffer = unit

    let alloc _dev _size _kind = ()

    let alloc_custom _dev ~size:_ ~elem_size:_ = ()

    let alloc_zero_copy _dev _ba _kind = None

    let alloc_host_ptr _dev _ba _kind = Error Framework_sig.Host_ptr_unsupported

    let device_ptr _buf = Nativeint.zero

    let is_zero_copy _buf = false

    let host_to_device ~src:_ ~dst:_ = ()

    let device_to_host ~src:_ ~dst:_ = ()

    let device_to_device ~src:_ ~dst:_ = ()

    let host_ptr_to_device ~src_ptr:_ ~byte_size:_ ~dst:_ = ()

    let device_to_host_ptr ~src:_ ~dst_ptr:_ ~byte_size:_ = ()

    let size _buf = 0

    let free _buf = ()
  end

  module Kernel = struct
    type t = unit

    type args = unit

    let compile _dev ~name:_ ~source:_ = ()

    let compile_cached _dev ~name:_ ~source:_ = ()

    let clear_cache () = ()

    let load_from_ptx ~name:_ ~ptx:_ = ()

    let create_args () = ()

    let set_arg_buffer _args _idx _buf = ()

    let set_arg_int32 _args _idx _v = ()

    let set_arg_int64 _args _idx _v = ()

    let set_arg_float32 _args _idx _v = ()

    let set_arg_float64 _args _idx _v = ()

    let set_arg_ptr _args _idx _ptr = ()

    let launch _kernel ~args:_ ~grid:_ ~block:_ ~shared_mem:_ ~stream:_ = ()
  end

  module Event = struct
    type t = unit

    let create () = ()

    let record _event _stream = ()

    let synchronize _event = ()

    let elapsed ~start:_ ~stop:_ = 0.0

    let destroy _event = ()
  end

  module Async = Spoc_framework.Framework_sig.Async_refused (struct
    let backend = name

    type device = Device.t

    type stream = Stream.t

    type event = Event.t

    type 'a buffer = 'a Memory.buffer
  end)

  let run_source_on_stream ~stream:_ ~source:_ ~lang:_ ~kernel_name:_ ~block:_
      ~grid:_ ~shared_mem:_ _args =
    Spoc_framework.Framework_sig.refuse_stream_launch ~backend:name

  include Spoc_framework.Framework_sig.Make_refs (struct
    type stream = Stream.t

    type event = Event.t
  end)

  let enable_profiling () = ()

  let disable_profiling () = ()
end

let () = Framework_registry.register_backend ~priority:1 (module Probe)

let caps () : capabilities =
  {
    max_threads_per_block = 256;
    max_block_dims = (256, 256, 64);
    max_grid_dims = (65535, 65535, 65535);
    shared_mem_per_block = 16384;
    total_global_mem = 1073741824L;
    compute_capability = (0, 0);
    device_features = [];
    coopmat = None;
    supports_atomics = false;
    warp_size = 32;
    max_registers_per_block = 0;
    clock_rate_khz = 0;
    multiprocessor_count = 1;
    is_cpu = false;
  }

let device ?(id = 0) ?(capabilities = caps ()) () : Device.t =
  {id; backend_id = 0; name = "probe"; framework = backend_name; capabilities}

(* kernel n -> ()   with one scalar parameter of type [ty]. A fresh record per
   call: two calls give structurally equal, physically distinct IRs. *)
let kernel ?(name = "memo_k") ty : Sarek_ir_types.kernel =
  let open Sarek_ir_types in
  let n = {var_name = "n"; var_id = 0; var_type = ty; var_mutable = false} in
  {default_kernel with kern_name = name; kern_params = [DParam (n, None)]}

let launch ?(block = 64) ~device ir args =
  Execute.run_vectors
    ~device
    ~ir
    ~args
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d 1)
    ()

let reset () =
  Execute.clear_launch_cache () ;
  Atomic.set codegen_calls 0 ;
  (codegen_hook := fun () -> ()) ;
  codegen_mode := `Ok ;
  launched := []

open Alcotest

(* AC-1 *)
let test_repeated_launch_generates_once () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  for i = 1 to 3 do
    launch ~device:dev ir [Execute.Int32 (Int32.of_int i)]
  done ;
  check int "generate_source calls" 1 (Atomic.get codegen_calls) ;
  match !launched with
  | [a; b; c] ->
      check bool "launches share one physical source" true (a == b && b == c)
  | l -> failf "expected 3 launches, got %d" (List.length l)

(* AC-2 *)
let test_distinct_ir_is_a_miss () =
  reset () ;
  let dev = device () in
  let ir1 = kernel Sarek_ir_types.TInt32
  and ir2 = kernel Sarek_ir_types.TInt32 in
  check bool "structurally equal" true (ir1 = ir2) ;
  launch ~device:dev ir1 [Execute.Int32 1l] ;
  launch ~device:dev ir2 [Execute.Int32 1l] ;
  check int "one codegen per physical IR" 2 (Atomic.get codegen_calls)

(* AC-3 (block part) *)
let test_block_is_part_of_the_key () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  launch ~block:64 ~device:dev ir [Execute.Int32 1l] ;
  launch ~block:128 ~device:dev ir [Execute.Int32 1l] ;
  launch ~block:64 ~device:dev ir [Execute.Int32 1l] ;
  check int "one codegen per block" 2 (Atomic.get codegen_calls) ;
  match !launched with
  | [s64b; s128; s64a] ->
      check bool "block 64 reused" true (s64a == s64b) ;
      check bool "block 128 differs" false (s128 == s64a)
  | _ -> fail "expected 3 launches"

(* AC-4: a refusal is raised on every launch, and a device carrying another
   capabilities record under the same id is re-checked. *)
let test_capability_refusal_every_launch () =
  reset () ;
  let ir = kernel Sarek_ir_types.TInt64 in
  let refused dev =
    match launch ~device:dev ir [Execute.Int64 1L] with
    | () -> false
    | exception Sarek_execute.Execute_error.Execution_error _ -> true
  in
  let poor = device () in
  check bool "refused (1st)" true (refused poor) ;
  check bool "refused (2nd)" true (refused poor) ;
  let rich =
    device
      ~capabilities:{(caps ()) with device_features = [Sarek_ir_analysis.Int64]}
      ()
  in
  check bool "accepted with int64" false (refused rich) ;
  (* Same id as [rich] (0), but the poor capabilities record: must not ride on
     [rich]'s cached verdict. *)
  check bool "refused again for the poor record" true (refused poor)

(* AC-5 *)
let test_arity_checked_after_a_hit () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  launch ~device:dev ir [Execute.Int32 1l] ;
  match launch ~device:dev ir [Execute.Int32 1l; Execute.Int32 2l] with
  | () -> fail "wrong arity accepted after a memo hit"
  | exception
      Sarek_execute.Execute_error.Execution_error
        (Sarek_execute.Execute_error.Type_mismatch _) ->
      ()

(* AC-6 *)
let test_failures_not_memoised () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  codegen_mode := `Raise ;
  for _ = 1 to 2 do
    match launch ~device:dev ir [Execute.Int32 1l] with
    | () -> fail "raising codegen was skipped"
    | exception Failure _ -> ()
  done ;
  codegen_mode := `None ;
  for _ = 1 to 2 do
    match launch ~device:dev ir [Execute.Int32 1l] with
    | () -> fail "None codegen was skipped"
    | exception Sarek_execute.Execute_error.Execution_error _ -> ()
  done ;
  check int "every failing launch ran codegen" 4 (Atomic.get codegen_calls) ;
  codegen_mode := `Ok ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "then succeeds and generates" 5 (Atomic.get codegen_calls)

(* AC-9 *)
let test_clear_and_hooks () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  launch ~device:dev ir [Execute.Int32 1l] ;
  Execute.clear_launch_cache () ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "clear_launch_cache forces codegen" 2 (Atomic.get codegen_calls) ;
  Cache_hooks.notify_clear_all () ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "notify_clear_all forces codegen" 3 (Atomic.get codegen_calls) ;
  Cache_hooks.notify_device_destroy ~backend:"HIP" 0 ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "device destroy forces codegen" 4 (Atomic.get codegen_calls) ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "and is memoised again" 4 (Atomic.get codegen_calls)

(* FR-005: the memo is weak in the IR. *)
let test_memo_does_not_retain_ir () =
  reset () ;
  let dev = device () in
  let w = Weak.create 1 in
  (let ir = kernel ~name:"ephemeral" Sarek_ir_types.TInt32 in
   Weak.set w 0 (Some ir) ;
   launch ~device:dev ir [Execute.Int32 1l]) ;
  Gc.full_major () ;
  Gc.full_major () ;
  check bool "IR collected after its last launch" false (Weak.check w 0)

(* EC-2: same IR on two devices of one backend: the source is device
   independent, so one codegen. *)
let test_two_devices_share_source () =
  reset () ;
  let caps = caps () in
  let d0 = device ~id:0 ~capabilities:caps ()
  and d1 = device ~id:1 ~capabilities:caps () in
  let ir = kernel Sarek_ir_types.TInt32 in
  launch ~device:d0 ir [Execute.Int32 1l] ;
  launch ~device:d1 ir [Execute.Int32 2l] ;
  check int "one codegen for both devices" 1 (Atomic.get codegen_calls)

(* FR-011: a clear that lands while a source is being generated must not let
   that source be installed afterwards. *)
let test_clear_during_codegen_not_installed () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  codegen_hook := Execute.clear_launch_cache ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  (codegen_hook := fun () -> ()) ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "regenerated after the racing clear" 2 (Atomic.get codegen_calls) ;
  launch ~device:dev ir [Execute.Int32 1l] ;
  check int "then memoised" 2 (Atomic.get codegen_calls)

(* R-41: one launch hashes its IR at most once, however many per-IR tables
   it consults (capability verdicts, source, launch window). Two kernels
   alternate, so no launch can reuse the previous launch's lookup. *)
let alternate_and_count ~device ~launches =
  let a = kernel ~name:"hash_a" Sarek_ir_types.TInt32
  and b = kernel ~name:"hash_b" Sarek_ir_types.TInt32 in
  launch ~device a [Execute.Int32 1l] ;
  launch ~device b [Execute.Int32 1l] ;
  Execute.Window.flush () ;
  let before = Sarek_execute.Ir_memo.hash_count () in
  for _ = 1 to launches / 2 do
    launch ~device a [Execute.Int32 1l] ;
    launch ~device b [Execute.Int32 1l]
  done ;
  let hashes = Sarek_execute.Ir_memo.hash_count () - before in
  Execute.Window.flush () ;
  hashes

let test_one_ir_hash_per_launch () =
  reset () ;
  let launches = 20 in
  let hashes = alternate_and_count ~device:(device ()) ~launches in
  check int "IR hashes for 20 alternating launches" launches hashes

let test_one_ir_hash_per_deferred_launch () =
  reset () ;
  Execute.Window.set_backend_enabled ~backend:backend_name true ;
  Fun.protect
    ~finally:(fun () ->
      Execute.Window.set_backend_enabled ~backend:backend_name false)
    (fun () ->
      let launches = 4 in
      let hashes = alternate_and_count ~device:(device ()) ~launches in
      check int "IR hashes for 4 alternating deferred launches" launches hashes)

(* Consecutive launches of one IR on one domain reuse that domain's last
   lookup: no hash at all. *)
let test_repeated_launch_hashes_nothing () =
  reset () ;
  let dev = device () and ir = kernel ~name:"hash_same" Sarek_ir_types.TInt32 in
  launch ~device:dev ir [Execute.Int32 1l] ;
  let before = Sarek_execute.Ir_memo.hash_count () in
  for i = 1 to 10 do
    launch ~device:dev ir [Execute.Int32 (Int32.of_int i)]
  done ;
  check
    int
    "IR hashes for 10 repeated launches"
    0
    (Sarek_execute.Ir_memo.hash_count () - before)

(* The launch-time matrix decision consults its per-IR plan once. *)
let test_one_ir_hash_per_matrix_decision () =
  reset () ;
  let hip = {(device ()) with framework = "HIP"} in
  let a = kernel ~name:"plan_a" Sarek_ir_types.TInt32
  and b = kernel ~name:"plan_b" Sarek_ir_types.TInt32 in
  let decide ir =
    match
      Sarek_execute.Matrix_specialise.choose
        ~device:hip
        ~ir
        ~launch:(fun () -> None)
        ()
    with
    | Sarek_execute.Matrix_specialise.Original _ -> ()
    | Specialised _ -> fail "a scalar kernel has no matrix product"
  in
  decide a ;
  decide b ;
  let before = Sarek_execute.Ir_memo.hash_count () in
  for _ = 1 to 5 do
    decide a ;
    decide b
  done ;
  check
    int
    "IR hashes for 10 alternating decisions"
    10
    (Sarek_execute.Ir_memo.hash_count () - before)

(* Replacing a backend under the same name (Framework_registry uses
   Hashtbl.replace) must not hand the replacement its predecessor's source. *)
module Probe_v2 : BACKEND = struct
  include Probe

  let generate_source ?block:_ ?soa_params:_ (_ : Sarek_ir_types.kernel) =
    Atomic.incr codegen_calls ;
    Some (String.concat "" ["// v2"; "\n"])
end

let test_replaced_backend_regenerates () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  launch ~device:dev ir [Execute.Int32 1l] ;
  Framework_registry.register_backend ~priority:1 (module Probe_v2) ;
  Fun.protect
    ~finally:(fun () ->
      Framework_registry.register_backend ~priority:1 (module Probe))
    (fun () ->
      launch ~device:dev ir [Execute.Int32 1l] ;
      check
        int
        "replacement generated its own source"
        2
        (Atomic.get codegen_calls) ;
      match !launched with
      | latest :: _ -> check string "replacement's source ran" "// v2\n" latest
      | [] -> fail "no launch recorded")

(* FR-006 / EC-4: concurrent launches from several domains all receive the one
   installed physical source. *)
let test_domains_share_one_source () =
  reset () ;
  let dev = device () and ir = kernel Sarek_ir_types.TInt32 in
  let per_domain = 200 in
  (* Force the race: the first two domains to miss wait for each other INSIDE
     codegen, so both build a source and the second install must find the
     first and hand it back. Bounded wait, so a lone miss cannot hang. *)
  let inside = Atomic.make 0 in
  (codegen_hook :=
     fun () ->
       Atomic.incr inside ;
       let deadline = Unix.gettimeofday () +. 2.0 in
       while Atomic.get inside < 2 && Unix.gettimeofday () < deadline do
         Domain.cpu_relax ()
       done) ;
  let worker () =
    for i = 1 to per_domain do
      launch ~device:dev ir [Execute.Int32 (Int32.of_int i)]
    done
  in
  let ds = List.init 4 (fun _ -> Domain.spawn worker) in
  List.iter Domain.join ds ;
  (codegen_hook := fun () -> ()) ;
  check
    bool
    "two domains really raced inside codegen"
    true
    (Atomic.get inside >= 2) ;
  let all = !launched in
  check int "every launch recorded" (4 * per_domain) (List.length all) ;
  check
    bool
    "at most one codegen per racing domain"
    true
    (Atomic.get codegen_calls <= 4) ;
  match all with
  | first :: rest ->
      check
        bool
        "one physical source for every launch"
        true
        (List.for_all (fun s -> s == first) rest)
  | [] -> fail "no launches"

(* The memoised source is keyed on the policy that decides its text
   (specs/optimisation-policy.md, M2): a grant given after a kernel's source
   was generated is honoured by the next launch. *)
let narrowing_kernel name : Sarek_ir_types.kernel =
  let open Sarek_ir_types in
  let var n ty =
    {var_name = n; var_id = 0; var_type = ty; var_mutable = false}
  in
  let vec n elt =
    DParam (var n (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})
  in
  let tid = var "tid" TInt32 in
  {
    default_kernel with
    kern_name = name;
    kern_params = [vec "out" TFloat16; vec "h" TFloat16];
    kern_body =
      SLet
        ( tid,
          EIntrinsic ([], "global_thread_id", []),
          SAssign
            ( LArrayElem ("out", EVar tid),
              ECast
                ( TFloat16,
                  EBinop
                    ( Mul,
                      ECast (TFloat32, EArrayRead ("h", EVar tid)),
                      EConst (CFloat32 1.1) ) ) ) );
  }

let has_barrier source =
  let needle = "__float2half(sarek_f32_barrier(" in
  let nl = String.length needle in
  let rec go i =
    i + nl <= String.length source
    && (String.sub source i nl = needle || go (i + 1))
  in
  go 0

let memo_source ir =
  match
    Execute.generate_source_memo
      (module Probe : BACKEND)
      ~block:(Execute.dims1d 64)
      ~soa_params:[]
      ir
  with
  | Some s -> s
  | None -> fail "no source"

let grant_single scope =
  match
    Sarek_policy.grant
      scope
      (Sarek_policy.Deviation Sarek_policy.Single_rounding)
      ~reason:"memo test"
  with
  | Ok () -> ()
  | Error _ -> fail "grant refused"

let test_grant_after_generation () =
  codegen_mode := `Cuda ;
  Fun.protect
    ~finally:(fun () -> codegen_mode := `Ok)
    (fun () ->
      let ir = narrowing_kernel "memo_stale_a" in
      let first = memo_source ir in
      check bool "barriers before the grant" true (has_barrier first) ;
      check
        bool
        "memo hit while the policy is unchanged"
        true
        (memo_source ir == first) ;
      grant_single (Sarek_policy.Kernel "memo_stale_a") ;
      let second = memo_source ir in
      check bool "no barrier after a Kernel grant" false (has_barrier second) ;
      check bool "and the new source is memoised" true (memo_source ir == second) ;
      let other = narrowing_kernel "memo_stale_b" in
      check
        bool
        "another kernel keeps its barriers"
        true
        (has_barrier (memo_source other)))

let test_unrelated_grant_keeps_source () =
  codegen_mode := `Cuda ;
  Fun.protect
    ~finally:(fun () -> codegen_mode := `Ok)
    (fun () ->
      let ir = narrowing_kernel "memo_stale_c" in
      let first = memo_source ir in
      grant_single (Sarek_policy.Kernel "memo_stale_unrelated") ;
      check
        bool
        "a grant for another kernel keeps the source"
        true
        (memo_source ir == first))

(* A launch under an unchanged policy computes no digest (one atomic read of
   the policy version); after a change elsewhere it computes one, and the stamp
   is refreshed. *)
let test_unchanged_policy_computes_no_digest () =
  codegen_mode := `Cuda ;
  Fun.protect
    ~finally:(fun () -> codegen_mode := `Ok)
    (fun () ->
      let ir = narrowing_kernel "memo_digest_a" in
      ignore (memo_source ir) ;
      let digests () = Sarek_execute.Policy_stamp.digests_computed () in
      let before = digests () in
      for _ = 1 to 50 do
        ignore (memo_source ir)
      done ;
      check int "no digest while the policy is unchanged" before (digests ()) ;
      grant_single (Sarek_policy.Kernel "memo_digest_unrelated") ;
      for _ = 1 to 50 do
        ignore (memo_source ir)
      done ;
      check int "one digest after a change elsewhere" (before + 1) (digests ()))

let () =
  run
    "Launch memo"
    [
      ( "memo",
        [
          test_case
            "repeated launch generates once"
            `Quick
            test_repeated_launch_generates_once;
          test_case "distinct IR is a miss" `Quick test_distinct_ir_is_a_miss;
          test_case
            "block is part of the key"
            `Quick
            test_block_is_part_of_the_key;
          test_case
            "capability refusal on every launch"
            `Quick
            test_capability_refusal_every_launch;
          test_case
            "arity checked after a hit"
            `Quick
            test_arity_checked_after_a_hit;
          test_case "failures not memoised" `Quick test_failures_not_memoised;
          test_case "clear and teardown hooks" `Quick test_clear_and_hooks;
          test_case
            "memo does not retain the IR"
            `Quick
            test_memo_does_not_retain_ir;
          test_case
            "two devices share one source"
            `Quick
            test_two_devices_share_source;
          test_case
            "clear during codegen is not installed"
            `Quick
            test_clear_during_codegen_not_installed;
          test_case
            "replaced backend regenerates"
            `Quick
            test_replaced_backend_regenerates;
          test_case
            "domains share one source"
            `Quick
            test_domains_share_one_source;
          test_case "one IR hash per launch" `Quick test_one_ir_hash_per_launch;
          test_case
            "repeated launch hashes nothing"
            `Quick
            test_repeated_launch_hashes_nothing;
          test_case
            "one IR hash per deferred launch"
            `Quick
            test_one_ir_hash_per_deferred_launch;
          test_case
            "one IR hash per matrix decision"
            `Quick
            test_one_ir_hash_per_matrix_decision;
          test_case
            "a grant after generation is honoured"
            `Quick
            test_grant_after_generation;
          test_case
            "an unrelated grant keeps the source"
            `Quick
            test_unrelated_grant_keeps_source;
          test_case
            "an unchanged policy computes no digest"
            `Quick
            test_unchanged_policy_computes_no_digest;
        ] );
    ]
