(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Launch window semantics (specs/launch-window.md, CHECK-1).
 *
 * A JIT probe backend logs every backend call it receives (launches with
 * their kernel name, grid and argument count, uploads, read-backs, device and
 * stream synchronisations, raw kernel launches), so each test asserts the ORDER
 * in which deferred launches reach the backend relative to the operation that
 * must observe them. Device memory is not real: these tests are about order,
 * fusion decisions and error handling; values are checked by
 * sarek/tests/e2e/test_launch_window.ml on real devices.
 ******************************************************************************)

open Spoc_framework
open Spoc_framework.Framework_sig
open Spoc_framework_registry
module Execute = Sarek_execute.Execute
module Device = Spoc_core.Device
module Vector = Spoc_core.Vector
module Transfer = Spoc_core.Transfer

let backend_name = "LaunchWindowProbe"

let log_lock = Mutex.create ()

let events : string list ref = ref []

let log e = Mutex.protect log_lock (fun () -> events := e :: !events)

let fail_names : string list ref = ref []

let fail_fused = ref false

let slow_started = Atomic.make false

let fail_epi = ref false

type Framework_sig.kargs += Probe_kargs

module Probe : BACKEND = struct
  let name = backend_name

  let version = (1, 0, 0)

  let is_available () = true

  let execution_model = JIT

  let supported_source_langs = [CUDA_Source]

  let generate_source ?block:_ ?soa_params:_ (ir : Sarek_ir_types.kernel) =
    Some ("// " ^ ir.Sarek_ir_types.kern_name)

  let execute_direct ~native_fn:_ ~ir:_ ~block:_ ~grid:_ _args = ()

  let wrap_kargs () = Probe_kargs

  let unwrap_kargs = function Probe_kargs -> Some () | _ -> None

  let run_source ~source:_ ~lang:_ ~kernel_name ~block:_ ~grid ~shared_mem:_
      args =
    if
      List.mem kernel_name !fail_names
      || (!fail_fused && String.starts_with ~prefix:"sarek_hfused" kernel_name)
      || (!fail_epi && String.starts_with ~prefix:"sarek_efused" kernel_name)
    then failwith ("probe launch refused: " ^ kernel_name) ;
    if kernel_name = "slow" then begin
      (* Tells the other domain the flush is under way, then takes its time:
         the launch is logged only when it is done. *)
      Atomic.set slow_started true ;
      Unix.sleepf 0.2
    end ;
    log
      (Printf.sprintf
         "launch %s gx=%d nargs=%d"
         kernel_name
         grid.x
         (List.length args))

  module Intrinsics = Intrinsic_registry.Make ()

  module Device = struct
    type t = unit

    type id = int

    let init () = ()

    let count () = 1

    let get _id = ()

    let id () = 0

    let set_current () = ()

    let synchronize () = log "devsync"

    let name () = "Launch window probe"

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

    let default () = ()

    let synchronize () = log "ssync"

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

    let host_ptr_to_device ~src_ptr:_ ~byte_size:_ ~dst:_ = log "h2d"

    let device_to_host_ptr ~src:_ ~dst_ptr:_ ~byte_size:_ = log "d2h"

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

    let launch _kernel ~args:_ ~grid:_ ~block:_ ~shared_mem:_ ~stream:_ =
      log "raw launch"
  end

  module Event = struct
    type t = unit

    let create () = ()

    let record _event _stream = log "event record"

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

let dev : Device.t =
  {
    id = 0;
    backend_id = 0;
    name = "probe";
    framework = backend_name;
    capabilities = caps ();
  }

let () = Execute.Window.set_backend_enabled ~backend:backend_name true

open Sarek_ir_types

let gpu name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

let fvar id name =
  {var_name = name; var_id = id; var_type = TVec TFloat32; var_mutable = false}

let arr = Some {arr_elttype = TFloat32; arr_memspace = Global}

(* [name src dst]: dst.(tid) <- src.(tid). A fresh IR per name. *)
let copy_kernel name =
  {
    default_kernel with
    kern_name = name;
    kern_params = [DParam (fvar 1 "src", arr); DParam (fvar 2 "dst", arr)];
    kern_body =
      SAssign
        ( LArrayElem ("dst", gpu "thread_idx_x"),
          EArrayRead ("src", gpu "thread_idx_x") );
  }

let k_a = copy_kernel "k_a"

let k_b = copy_kernel "k_b"

let k_c = copy_kernel "k_c"

let vec () = Vector.create Vector.float32 8

let launch ?(gx = 1) ?(block = 64) ir src dst =
  Execute.run_vectors
    ~device:dev
    ~ir
    ~args:[Execute.Vec src; Execute.Vec dst]
    ~block:(Execute.dims1d block)
    ~grid:(Execute.dims1d gx)
    ()

(* Fused kernels are named sarek_hfused_<digest of their IR>: the suffix is
   replaced by N so expectations do not spell digests. *)
let normalise e =
  String.split_on_char ' ' e
  |> List.map (fun w ->
      if String.starts_with ~prefix:"sarek_hfused_" w then "sarek_hfused_N"
      else w)
  |> String.concat " "

let launches () =
  List.filter_map
    (fun e ->
      if String.starts_with ~prefix:"launch" e then Some (normalise e) else None)
    (List.rev !events)

let reset () =
  Execute.Window.flush () ;
  Execute.Window.reset_stats () ;
  Execute.Window.clear_fused_cache () ;
  fail_names := [] ;
  fail_fused := false ;
  events := []

(* Vectors already resident on the probe device, so later launches do not
   upload them. *)
let resident n =
  List.init n (fun _ ->
      let v = vec () in
      Transfer.to_device v dev ;
      v)

let index_of e =
  let rec go i = function
    | [] -> Alcotest.failf "event %S not logged" e
    | x :: rest -> if String.starts_with ~prefix:e x then i else go (i + 1) rest
  in
  go 0 (List.rev !events)

open Alcotest

let test_deferred_until_flush () =
  reset () ;
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  launch k_a x y ;
  check (list string) "nothing issued yet" [] (launches ()) ;
  Execute.Window.flush () ;
  check
    (list string)
    "issued at flush"
    ["launch k_a gx=1 nargs=4"]
    (launches ())

let test_independent_fused () =
  reset () ;
  let[@warning "-8"] [x; y1; y2; y3] = resident 4 in
  events := [] ;
  launch ~gx:3 k_a x y1 ;
  launch ~gx:2 k_b x y2 ;
  launch ~gx:5 k_a x y3 ;
  Execute.Window.flush () ;
  (* 3 members x 4 args + 3 x (off, gdx) = 18 *)
  check
    (list string)
    "one fused launch over 10 groups"
    ["launch sarek_hfused_N gx=10 nargs=18"]
    (launches ()) ;
  let s = Execute.Window.stats () in
  check int "submitted" 3 s.submitted ;
  check int "issued" 1 s.issued ;
  check int "fused groups" 1 s.fused_groups

let test_dependent_in_order () =
  reset () ;
  let[@warning "-8"] [x; t; y] = resident 3 in
  events := [] ;
  launch k_a x t ;
  launch k_b t y ;
  (* RAW on t *)
  launch k_c y x ;
  (* WAR on x (read by k_a), RAW on y *)
  Execute.Window.flush () ;
  check
    (list string)
    "three launches, in order"
    [
      "launch k_a gx=1 nargs=4";
      "launch k_b gx=1 nargs=4";
      "launch k_c gx=1 nargs=4";
    ]
    (launches ())

let test_reordered_past_independent () =
  reset () ;
  let[@warning "-8"] [x; t; y; z] = resident 4 in
  events := [] ;
  launch k_a x t ;
  launch k_b t y ;
  (* depends on k_a *)
  launch k_c x z ;
  (* independent of both: joins... the newest compatible group is k_b's *)
  Execute.Window.flush () ;
  check
    (list string)
    "k_c fused with k_b, after k_a"
    ["launch k_a gx=1 nargs=4"; "launch sarek_hfused_N gx=2 nargs=12"]
    (launches ())

let test_different_block_not_fused () =
  reset () ;
  let[@warning "-8"] [x; y1; y2] = resident 3 in
  events := [] ;
  launch ~block:64 k_a x y1 ;
  launch ~block:128 k_b x y2 ;
  Execute.Window.flush () ;
  check int "two launches" 2 (List.length (launches ()))

(* Each synchronisation point issues the pending launch before it acts: the
   launch is logged before the operation's own backend call ([marker]). [op]
   receives the launch's output vector. *)
let sync_point what ~marker op () =
  reset () ;
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  launch k_a x y ;
  op y ;
  check
    bool
    (what ^ ": launch logged before " ^ marker)
    true
    (index_of "launch k_a" < index_of marker)

let raw_kernel_launch _ =
  let k = Spoc_core.Kernel.compile dev ~name:"raw" ~source:"" in
  Spoc_core.Kernel.launch
    k
    ~args:(Spoc_core.Kernel.create_args dev)
    ~grid:(Execute.dims1d 1)
    ~block:(Execute.dims1d 1)
    ()

let sync_points =
  [
    ("Transfer.flush", "devsync", fun _ -> Transfer.flush dev);
    ("Device.synchronize", "devsync", fun _ -> Device.synchronize dev);
    ("Transfer.to_cpu", "d2h", fun y -> Transfer.to_cpu y);
    ("Vector.get (auto-sync)", "d2h", fun y -> ignore (Vector.get y 0));
    ("Transfer.free_buffer", "d2h", fun y -> Transfer.free_buffer y dev);
    ( "default stream sync",
      "ssync",
      fun _ -> Transfer.synchronize_stream (Transfer.default_stream dev) );
    ("Kernel.launch (raw)", "raw launch", raw_kernel_launch);
    ( "Execute.run_source",
      "launch ext",
      fun y ->
        Execute.run_source
          ~device:dev
          ~source:""
          ~lang:Execute.CUDA_Source
          ~kernel_name:"ext"
          ~block:(Execute.dims1d 1)
          ~grid:(Execute.dims1d 1)
          [Execute.Vec y] );
  ]

(* An upload into a buffer a pending launch binds issues that launch first;
   an upload into an unrelated buffer does not. *)
let test_upload_touching_flushes () =
  reset () ;
  let[@warning "-8"] [x; y; z] = resident 3 in
  events := [] ;
  launch k_a x y ;
  Vector.set x 0 1.0 ;
  (* x is now newer on the host: the next launch uploads it *)
  launch k_b x z ;
  check
    bool
    "k_a issued before the upload of x"
    true
    (index_of "launch k_a" < index_of "h2d")

let test_upload_unrelated_keeps_pending () =
  reset () ;
  let[@warning "-8"] [x; y; z] = resident 3 in
  events := [] ;
  launch k_a x y ;
  let fresh = vec () in
  launch k_b fresh z ;
  (* uploads [fresh] *)
  check (list string) "nothing issued by the unrelated upload" [] (launches ()) ;
  Execute.Window.flush () ;
  check
    (list string)
    "both fused"
    ["launch sarek_hfused_N gx=2 nargs=12"]
    (launches ())

let test_issue_error_reraised_after_rest () =
  reset () ;
  let[@warning "-8"] [x; t; y; z] = resident 4 in
  events := [] ;
  fail_names := ["k_b"] ;
  launch k_a x t ;
  launch k_b t y ;
  launch k_c y z ;
  (match Transfer.flush dev with
  | () -> fail "the failed launch must be reported by the flushing call"
  | exception Failure _ -> ()) ;
  check
    (list string)
    "k_a and k_c issued"
    ["launch k_a gx=1 nargs=4"; "launch k_c gx=1 nargs=4"]
    (launches ())

let test_fused_failure_falls_back () =
  reset () ;
  let[@warning "-8"] [x; y1; y2] = resident 3 in
  events := [] ;
  fail_fused := true ;
  let pair () =
    launch k_a x y1 ;
    launch k_b x y2 ;
    Execute.Window.flush ()
  in
  pair () ;
  check
    (list string)
    "members issued one by one"
    ["launch k_a gx=1 nargs=4"; "launch k_b gx=1 nargs=4"]
    (launches ()) ;
  check int "refused once" 1 (Execute.Window.stats ()).fused_refused ;
  events := [] ;
  pair () ;
  check
    (list string)
    "not fused again"
    ["launch k_a gx=1 nargs=4"; "launch k_b gx=1 nargs=4"]
    (launches ()) ;
  check int "no second attempt" 1 (Execute.Window.stats ()).fused_refused

let test_disabled_is_immediate () =
  reset () ;
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  Execute.Window.set_enabled false ;
  Fun.protect
    ~finally:(fun () -> Execute.Window.set_enabled true)
    (fun () ->
      launch k_a x y ;
      check
        (list string)
        "issued in the call"
        ["launch k_a gx=1 nargs=4"]
        (launches ()))

let test_window_is_bounded () =
  reset () ;
  let vs = resident 10 in
  events := [] ;
  (* A chain: each launch reads the previous one's output, so none fuse. *)
  List.iteri (fun i v -> if i > 0 then launch k_a (List.nth vs (i - 1)) v) vs ;
  check
    int
    "oldest issued once 9 groups are pending"
    1
    (List.length (launches ())) ;
  Execute.Window.flush () ;
  check int "all issued" 9 (List.length (launches ()))

let test_domains () =
  reset () ;
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  (* Every launch writes y: no two may fuse, so each is one backend launch. *)
  let worker () =
    for i = 1 to 50 do
      launch k_a x y ;
      if i mod 7 = 0 then Execute.Window.flush ()
    done
  in
  let ds = List.init 4 (fun _ -> Domain.spawn worker) in
  List.iter Domain.join ds ;
  Execute.Window.flush () ;
  check int "every launch issued exactly once" 200 (List.length (launches ()))

(* Synchronisation points whose probe call may refuse (no async support on the
   probe): the pending launch must have been issued before the call acts,
   whatever it then does. *)
let issued_by what op () =
  reset () ;
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  launch k_a x y ;
  (try op y with _ -> ()) ;
  check
    bool
    (what ^ ": pending launch issued")
    true
    (List.mem "launch k_a gx=1 nargs=4" (launches ()))

let small = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 2

let more_sync_points =
  [
    ( "Profiling timer",
      fun _ ->
        let t = Spoc_core.Profiling.create_timer dev in
        Spoc_core.Profiling.timer_start t );
    ( "Transfer.record_event",
      fun _ ->
        let ev = Transfer.create_event dev in
        Transfer.record_event ev (Transfer.default_stream dev) );
    ( "Transfer.copy_to_device",
      fun y -> Transfer.copy_to_device small ~dst:y ~dst_offset:0 );
    ( "Transfer.copy_to_host",
      fun y -> Transfer.copy_to_host ~src:y ~src_offset:0 small );
    ( "Transfer.stage_to_device",
      fun y -> Transfer.stage_to_device small ~dst:y ~dst_offset:0 );
    ("Transfer.register_host", fun _ -> Transfer.register_host dev small);
    ("cache clear", fun _ -> Spoc_framework.Cache_hooks.notify_clear_all ());
    ( "device destroy",
      fun _ ->
        Spoc_framework.Cache_hooks.notify_device_destroy ~backend:backend_name 0
    );
  ]

(* {1 IR analysis and rewrite} *)

let ivar id name =
  {var_name = name; var_id = id; var_type = TInt32; var_mutable = false}

let verdict body
    ?(params = [DParam (fvar 1 "src", arr); DParam (fvar 2 "dst", arr)]) () =
  Sarek_execute.Kernel_access.horizontal_verdict
    {
      default_kernel with
      kern_name = "v";
      kern_params = params;
      kern_body = body;
    }

let refused why body () =
  match verdict body () with
  | Ok _ -> failf "%s: expected a refusal" why
  | Error _ -> ()

let store e = SAssign (LArrayElem ("dst", gpu "thread_idx_x"), e)

let test_verdicts () =
  refused "grid_dim_x" (store (ECast (TFloat32, gpu "grid_dim_x"))) () ;
  refused
    "block_id_x alias"
    (store
       (ECast (TFloat32, EIntrinsic (["Sarek_stdlib"; "Gpu"], "block_id_x", []))))
    () ;
  refused
    "block_idx_x under another path"
    (store
       (ECast
          (TFloat32, EIntrinsic (["Sarek_stdlib"; "Other"], "block_idx_x", []))))
    () ;
  refused
    "downstream intrinsic"
    (store
       (EIntrinsic (["My_lib"], "f", [EArrayRead ("src", gpu "thread_idx_x")])))
    () ;
  refused "free variable" (store (ECast (TFloat32, EVar (ivar 99 "g")))) () ;
  refused
    "shared array inside the body"
    (SSeq
       [
         SEmpty;
         SLet
           ( fvar 7 "s",
             EArrayCreate (TFloat32, EConst (CInt32 4l), Shared),
             SEmpty );
       ])
    () ;
  refused
    "shared array of non-constant size"
    (SLet
       (fvar 7 "s", EArrayCreate (TFloat32, gpu "block_dim_x", Shared), SEmpty))
    () ;
  match
    verdict
      (SLet
         ( fvar 7 "s",
           EArrayCreate (TFloat32, EConst (CInt32 64l), Shared),
           store (EArrayRead ("src", gpu "thread_idx_x")) ))
      ()
  with
  | Ok bytes -> check int "shared bytes counted" 256 bytes
  | Error why -> failf "plain kernel refused: %s" why

let accesses body =
  Sarek_execute.Kernel_access.param_accesses
    {
      default_kernel with
      kern_name = "a";
      kern_params = [DParam (fvar 1 "src", arr); DParam (fvar 2 "dst", arr)];
      kern_body = body;
    }

let rw body name =
  let a = List.assoc name (accesses body) in
  (a.Sarek_execute.Kernel_access.read, a.write)

let test_accesses () =
  let tid = gpu "thread_idx_x" in
  check
    (pair bool bool)
    "plain read"
    (true, false)
    (rw (store (EArrayRead ("src", tid))) "src") ;
  check
    (pair bool bool)
    "plain write"
    (false, true)
    (rw (store (EArrayRead ("src", tid))) "dst") ;
  check
    (pair bool bool)
    "vector passed whole (alias, helper, intrinsic)"
    (true, true)
    (rw (SLet (fvar 9 "alias", EVar (fvar 1 "src"), SEmpty)) "src") ;
  check
    (pair bool bool)
    "atomic on an element"
    (true, true)
    (rw
       (SExpr
          (EIntrinsic
             ( ["Sarek_stdlib"; "Gpu"],
               "atomic_add_int32",
               [EArrayRead ("src", tid); EConst (CInt32 1l)] )))
       "src") ;
  check
    (pair bool bool)
    "field of an element"
    (false, true)
    (rw
       (SAssign
          (LRecordField (LArrayElem ("dst", tid), "f"), EConst (CFloat32 0.)))
       "dst")

(* A member local named like the position parameters must not capture them
   (review finding: a local [off] became [m0_off] and shadowed the
   parameter). *)
let test_rename_no_capture () =
  let member =
    {
      default_kernel with
      kern_name = "m";
      kern_params = [DParam (fvar 1 "dst", arr)];
      kern_body =
        SLet
          ( ivar 5 "off",
            EConst (CInt32 7l),
            SLet
              ( ivar 6 "gdx",
                EConst (CInt32 9l),
                SAssign
                  ( LArrayElem ("dst", gpu "global_thread_id"),
                    ECast
                      ( TFloat32,
                        EBinop (Add, EVar (ivar 5 "off"), EVar (ivar 6 "gdx"))
                      ) ) ) );
    }
  in
  let fused = Sarek_execute.Fusion_horizontal.fuse ~name:"f" [member; member] in
  let names =
    List.filter_map
      (function DParam (v, _) -> Some v.var_name | _ -> None)
      fused.kern_params
  in
  check
    (list string)
    "parameters"
    ["m0_dst"; "m1_dst"; "hf0_off"; "hf0_gdx"; "hf1_off"; "hf1_gdx"]
    names ;
  let src = Sarek_codegen.Sarek_ir_cuda.generate fused in
  let contains sub =
    let n = String.length sub and m = String.length src in
    let rec go i = i + n <= m && (String.sub src i n = sub || go (i + 1)) in
    go 0
  in
  check
    bool
    "position rewritten with the parameter"
    true
    (contains "((int)blockIdx.x) - hf0_off") ;
  check bool "member local keeps its own name" true (contains "int m0_off = 7")

(* The fused grid must fit the device's grid limit. *)
let test_grid_limit () =
  reset () ;
  let small_dev =
    {dev with capabilities = {(caps ()) with max_grid_dims = (4, 65535, 65535)}}
  in
  let[@warning "-8"] [x; y1; y2] = resident 3 in
  events := [] ;
  let go ir y gx =
    Execute.run_vectors
      ~device:small_dev
      ~ir
      ~args:[Execute.Vec x; Execute.Vec y]
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d gx)
      ()
  in
  go k_a y1 3 ;
  go k_b y2 3 ;
  Execute.Window.flush () ;
  check
    int
    "3 + 3 work-groups exceed a limit of 4: not fused"
    2
    (List.length (launches ()))

(* {1 Review round 3} *)

let test_cpu_device_not_deferred () =
  reset () ;
  let cpu = {dev with capabilities = {(caps ()) with is_cpu = true}} in
  let[@warning "-8"] [x; y] = resident 2 in
  events := [] ;
  Execute.run_vectors
    ~device:cpu
    ~ir:k_a
    ~args:[Execute.Vec x; Execute.Vec y]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    () ;
  check
    (list string)
    "issued in the call (zero-copy buffers may alias)"
    ["launch k_a gx=1 nargs=4"]
    (launches ())

(* [name src dst] with a 64-float shared array (256 bytes). *)
let shared_kernel name =
  let s = fvar 7 "s" in
  {
    (copy_kernel name) with
    kern_body =
      SLet
        ( s,
          EArrayCreate (TFloat32, EConst (CInt32 64l), Shared),
          SAssign
            ( LArrayElem ("dst", gpu "thread_idx_x"),
              EArrayRead ("src", gpu "thread_idx_x") ) );
  }

let s_a = shared_kernel "s_a"

let s_b = shared_kernel "s_b"

let test_shared_limit () =
  let pair limit =
    reset () ;
    let d =
      {dev with capabilities = {(caps ()) with shared_mem_per_block = limit}}
    in
    let[@warning "-8"] [x; y1; y2] = resident 3 in
    events := [] ;
    let go ir y =
      Execute.run_vectors
        ~device:d
        ~ir
        ~args:[Execute.Vec x; Execute.Vec y]
        ~block:(Execute.dims1d 64)
        ~grid:(Execute.dims1d 1)
        ()
    in
    go s_a y1 ;
    go s_b y2 ;
    Execute.Window.flush () ;
    List.length (launches ())
  in
  check int "256 + 256 bytes within 512: fused" 1 (pair 512) ;
  check int "256 + 256 bytes over 300: not fused" 2 (pair 300)

(* The oldest group fails when the window overflows: the launching call
   raises and its own launch is not made. *)
let test_overflow_failure () =
  reset () ;
  let vs = resident 10 in
  events := [] ;
  fail_names := ["k_b"] ;
  let chain i v =
    if i > 0 then launch (if i = 1 then k_b else k_a) (List.nth vs (i - 1)) v
  in
  (* Groups 1..8 pending; the 9th launch (i = 9) issues the oldest, k_b. *)
  for i = 0 to 8 do
    chain i (List.nth vs i)
  done ;
  (match chain 9 (List.nth vs 9) with
  | () -> fail "the overflow failure must be raised by the launching call"
  | exception Failure _ -> ()) ;
  fail_names := [] ;
  Execute.Window.flush () ;
  check
    int
    "7 launches of the chain issued, not the rejected one"
    7
    (List.length
       (List.filter (fun e -> e = "launch k_a gx=1 nargs=4") (launches ())))

let test_fused_fallback_member_fails () =
  reset () ;
  let[@warning "-8"] [x; y1; y2; y3] = resident 4 in
  events := [] ;
  fail_fused := true ;
  fail_names := ["k_b"] ;
  launch k_a x y1 ;
  launch k_b x y2 ;
  launch k_c x y3 ;
  (match Execute.Window.flush () with
  | () -> fail "the member's failure must be raised"
  | exception Failure _ -> ()) ;
  check
    (list string)
    "the members around the failing one are issued"
    ["launch k_a gx=1 nargs=4"; "launch k_c gx=1 nargs=4"]
    (launches ())

(* Argument expansion happens in the launching call: an argument with no
   device buffer is refused there, not at the next flush. *)
let test_argument_error_at_submit () =
  reset () ;
  let[@warning "-8"] [x] = resident 1 in
  let fresh = vec () in
  match
    Execute.run
      ~device:dev
      ~name:"k_a"
      ~ir:(Some (lazy k_a))
      ~native_fn:None
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d 1)
      [Execute.Vec x; Execute.Vec fresh]
  with
  | () -> fail "a vector with no device buffer must be refused in the call"
  | exception Sarek_execute.Execute_error.Execution_error _ ->
      check
        int
        "nothing pending"
        0
        (Atomic.get Spoc_framework.Launch_hooks.pending)

(* A synchronisation point on another domain waits for a flush in progress
   (review round 1, S1). *)
let test_sync_waits_for_flush_in_progress () =
  reset () ;
  let slow = copy_kernel "slow" in
  let[@warning "-8"] [x; y; z] = resident 3 in
  events := [] ;
  Atomic.set slow_started false ;
  launch slow x y ;
  let a = Domain.spawn (fun () -> Execute.Window.flush ()) in
  while not (Atomic.get slow_started) do
    Domain.cpu_relax ()
  done ;
  (* [z] is untouched: only the flush in progress can delay this. *)
  Transfer.flush dev ;
  Domain.join a ;
  ignore z ;
  check
    bool
    "the other domain's sync point came after the launch"
    true
    (index_of "launch slow" < index_of "devsync")

(* {1 Epilogue fusion (specs/epilogue-fusion.md)} *)

(* dst.(tid) <- src.(tid), only for tid < n: a producer storing [0, n). *)
let guarded_copy name =
  let n = ivar 3 "n" in
  {
    (copy_kernel name) with
    kern_params =
      [DParam (fvar 1 "src", arr); DParam (fvar 2 "dst", arr); DParam (n, None)];
    kern_body =
      SIf
        ( EBinop (Lt, gpu "thread_idx_x", EVar n),
          SAssign
            ( LArrayElem ("dst", gpu "thread_idx_x"),
              EArrayRead ("src", gpu "thread_idx_x") ),
          None );
  }

(* let i = global_thread_id in if i < n then y.(i) <- y.(i) +. t.(i) *)
let add_consumer name =
  let i = ivar 10 "i" and n = ivar 11 "n" in
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [DParam (fvar 12 "y", arr); DParam (fvar 13 "t", arr); DParam (n, None)];
    kern_body =
      SLet
        ( i,
          gpu "global_thread_id",
          SIf
            ( EBinop (Lt, EVar i, EVar n),
              SAssign
                ( LArrayElem ("y", EVar i),
                  EBinop
                    (Add, EArrayRead ("y", EVar i), EArrayRead ("t", EVar i)) ),
              None ) );
  }

let k_prod = guarded_copy "k_prod"

let k_cons = add_consumer "k_cons"

let test_index_slice () =
  let stored = ref [] in
  let used =
    Sarek_execute.Index_slice.run
      ~is_target:(fun a -> a = "dst")
      ~record:(fun i -> stored := Int32.to_int i :: !stored)
      ~scalars:[("n", 5l)]
      ~dims:{Sarek_execute.Index_slice.block = (8, 1, 1); grid = (1, 1, 1)}
      k_prod.kern_body
  in
  check (list int) "stores 0..4" [0; 1; 2; 3; 4] (List.sort compare !stored) ;
  check (list string) "read scalar n" ["n"] used ;
  let data_dependent =
    {
      k_prod with
      kern_body =
        SAssign
          ( LArrayElem
              ("dst", ECast (TInt32, EArrayRead ("src", gpu "thread_idx_x"))),
            EConst (CFloat32 0.) );
    }
  in
  match
    Sarek_execute.Index_slice.run
      ~is_target:(fun a -> a = "dst")
      ~record:ignore
      ~scalars:[]
      ~dims:{Sarek_execute.Index_slice.block = (8, 1, 1); grid = (1, 1, 1)}
      data_dependent.kern_body
  with
  | _ -> fail "a data-dependent index must be unknown"
  | exception Sarek_execute.Index_slice.Unknown _ -> ()

let test_consumer_shape () =
  (match Sarek_execute.Fusion_epilogue.consumer k_cons with
  | Ok c ->
      check (list string) "writes" ["y"] c.written ;
      check bool "reads t at its index" true (List.mem "t" c.reads_at_index)
  | Error why -> failf "add consumer refused: %s" why) ;
  let refused why k =
    match Sarek_execute.Fusion_epilogue.consumer k with
    | Ok _ -> failf "%s: expected a refusal" why
    | Error _ -> ()
  in
  let body_of k = match k.kern_body with SLet (_, _, b) -> b | b -> b in
  let with_body b =
    {k_cons with kern_body = SLet (ivar 10 "i", gpu "global_thread_id", b)}
  in
  refused "barrier" (with_body (SSeq [SBarrier; body_of k_cons])) ;
  refused
    "store off the index"
    (with_body
       (SAssign (LArrayElem ("y", EConst (CInt32 0l)), EConst (CFloat32 1.)))) ;
  refused
    "position intrinsic in the body"
    (with_body
       (SAssign
          ( LArrayElem ("y", EVar (ivar 10 "i")),
            ECast (TFloat32, gpu "thread_idx_x") ))) ;
  refused "not element-wise" k_prod

let epilogue_pair ?(fresh = true) ~cons_block () =
  if fresh then reset () else Execute.Window.flush () ;
  Execute.Window.set_epilogue_backend ~backend:backend_name true ;
  let[@warning "-8"] [x; t; y] = resident 3 in
  events := [] ;
  Execute.run_vectors
    ~device:dev
    ~ir:k_prod
    ~args:[Execute.Vec x; Execute.Vec t; Execute.Int 64]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    () ;
  Execute.run_vectors
    ~device:dev
    ~ir:k_cons
    ~args:[Execute.Vec y; Execute.Vec t; Execute.Int cons_block]
    ~block:(Execute.dims1d cons_block)
    ~grid:(Execute.dims1d 1)
    () ;
  Execute.Window.flush () ;
  Execute.Window.set_epilogue_backend ~backend:backend_name false ;
  launches ()

let test_epilogue_merged () =
  match epilogue_pair ~cons_block:64 () with
  | [l] ->
      check
        bool
        "one merged launch"
        true
        (String.starts_with ~prefix:"launch sarek_efused_" l)
  | ls -> failf "expected one merged launch, got %d" (List.length ls)

let test_epilogue_uncovered () =
  (* The consumer writes [0, 128) but the producer stores only [0, 64). *)
  check
    (list string)
    "not merged"
    ["launch k_prod gx=1 nargs=5"; "launch k_cons gx=1 nargs=5"]
    (epilogue_pair ~cons_block:128 ())

let test_epilogue_failure_falls_back () =
  fail_epi := true ;
  let ls = epilogue_pair ~cons_block:64 () in
  fail_epi := false ;
  check
    (list string)
    "producer then consumer, as submitted"
    ["launch k_prod gx=1 nargs=5"; "launch k_cons gx=1 nargs=5"]
    ls ;
  check
    (list string)
    "never merged again"
    ["launch k_prod gx=1 nargs=5"; "launch k_cons gx=1 nargs=5"]
    (epilogue_pair ~fresh:false ~cons_block:64 ())

let test_epilogue_off_by_default () =
  reset () ;
  check
    bool
    "probe backend off by default"
    false
    (Execute.Window.epilogue_on backend_name) ;
  check bool "HIP on by default" true (Execute.Window.epilogue_on "HIP")

(* {2 Review round (epilogue)} *)

let tid = gpu "thread_idx_x"

let store_at arr idx v = SAssign (LArrayElem (arr, idx), v)

(* a.(tid) <- src (tid < n); b.(tid) <- src (tid < m). *)
let two_out =
  let n = ivar 3 "n" and m = ivar 4 "m" in
  {
    default_kernel with
    kern_name = "two_out";
    kern_params =
      [
        DParam (fvar 1 "src", arr);
        DParam (fvar 5 "a", arr);
        DParam (fvar 6 "b", arr);
        DParam (n, None);
        DParam (m, None);
      ];
    kern_body =
      SSeq
        [
          SIf
            ( EBinop (Lt, tid, EVar n),
              store_at "a" tid (EArrayRead ("src", tid)),
              None );
          SIf
            ( EBinop (Lt, tid, EVar m),
              store_at "b" tid (EArrayRead ("src", tid)),
              None );
        ];
  }

(* The coverage verdict depends on which output is consumed (review B1). *)
let test_coverage_per_array () =
  reset () ;
  Execute.Window.set_epilogue_backend ~backend:backend_name true ;
  let pair consumed =
    let[@warning "-8"] [x; a; b; y] = resident 4 in
    events := [] ;
    Execute.run_vectors
      ~device:dev
      ~ir:two_out
      ~args:
        [
          Execute.Vec x;
          Execute.Vec a;
          Execute.Vec b;
          Execute.Int 64;
          Execute.Int 32;
        ]
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d 1)
      () ;
    let t = if consumed = "a" then a else b in
    Execute.run_vectors
      ~device:dev
      ~ir:k_cons
      ~args:[Execute.Vec y; Execute.Vec t; Execute.Int 64]
      ~block:(Execute.dims1d 64)
      ~grid:(Execute.dims1d 1)
      () ;
    Execute.Window.flush () ;
    List.length (launches ())
  in
  check int "a is stored on [0, 64): merged" 1 (pair "a") ;
  check int "b only on [0, 32): not merged" 2 (pair "b") ;
  Execute.Window.set_epilogue_backend ~backend:backend_name false

(* Each element stored twice: not merged. *)
let test_duplicate_stores () =
  let halves =
    {
      k_prod with
      kern_body =
        store_at
          "dst"
          (EBinop (Div, tid, EConst (CInt32 2l)))
          (EArrayRead ("src", tid));
    }
  in
  reset () ;
  Execute.Window.set_epilogue_backend ~backend:backend_name true ;
  let[@warning "-8"] [x; t; y] = resident 3 in
  events := [] ;
  Execute.run_vectors
    ~device:dev
    ~ir:halves
    ~args:[Execute.Vec x; Execute.Vec t; Execute.Int 64]
    ~block:(Execute.dims1d 64)
    ~grid:(Execute.dims1d 1)
    () ;
  Execute.run_vectors
    ~device:dev
    ~ir:k_cons
    ~args:[Execute.Vec y; Execute.Vec t; Execute.Int 32]
    ~block:(Execute.dims1d 32)
    ~grid:(Execute.dims1d 1)
    () ;
  Execute.Window.flush () ;
  Execute.Window.set_epilogue_backend ~backend:backend_name false ;
  check int "two launches" 2 (List.length (launches ()))

let consumer_of name ~src ~dst ~add =
  let i = ivar 20 "i" and n = ivar 21 "n" in
  {
    default_kernel with
    kern_name = name;
    kern_params =
      [DParam (fvar 22 dst, arr); DParam (fvar 23 src, arr); DParam (n, None)];
    kern_body =
      SLet
        ( i,
          gpu "global_thread_id",
          SIf
            ( EBinop (Lt, EVar i, EVar n),
              store_at
                dst
                (EVar i)
                (EBinop (Add, EArrayRead (src, EVar i), EConst (CFloat32 add))),
              None ) );
  }

let fbits v =
  Array.init (Vector.length v) (fun i -> Int32.bits_of_float (Vector.get v i))

(* A chain P -> C1 -> C2 run on the Interpreter, which keys variables by id
   (review S1: C1's guard used to read C2's [tc]). *)
let test_chain_on_interpreter () =
  let c1 = consumer_of "c1" ~src:"dst" ~dst:"o1" ~add:1.0
  and c2 = consumer_of "c2" ~src:"o1" ~dst:"o2" ~add:100.0 in
  let fuse = Sarek_execute.Fusion_epilogue.fuse in
  let pc1 =
    fuse
      ~name:"pc1"
      ~producer:k_prod
      ~p_array:"dst"
      ~consumer:c1
      ~shared:[("dst", "dst")]
  in
  let pc12 =
    fuse
      ~name:"pc12"
      ~producer:pc1
      ~p_array:"c_o1"
      ~consumer:c2
      ~shared:[("o1", "c_o1")]
  in
  let zeros () =
    let v = vec () in
    for i = 0 to Vector.length v - 1 do
      Vector.set v i 0.0
    done ;
    v
  in
  let mk () = (zeros (), zeros (), zeros (), zeros ()) in
  let fill v =
    for i = 0 to Vector.length v - 1 do
      Vector.set v i (float_of_int i)
    done
  in
  let interp ir args =
    Execute.run_interpreter_vectors
      ~ir
      ~args
      ~block:(Execute.dims1d 8)
      ~grid:(Execute.dims1d 1)
      ~parallel:false
  in
  let src, d, o1, o2 = mk () in
  fill src ;
  interp k_prod [Execute.Vec src; Execute.Vec d; Execute.Int 8] ;
  interp c1 [Execute.Vec o1; Execute.Vec d; Execute.Int 8] ;
  interp c2 [Execute.Vec o2; Execute.Vec o1; Execute.Int 4] ;
  let src', d', o1', o2' = mk () in
  fill src' ;
  (* pc12 params: P's (src, dst, n), C1's unshared (o1, n), tc1, then C2's
     unshared (o2, n), tc2. *)
  interp
    pc12
    [
      Execute.Vec src';
      Execute.Vec d';
      Execute.Int 8;
      Execute.Vec o1';
      Execute.Int 8;
      Execute.Int 8;
      Execute.Vec o2';
      Execute.Int 4;
      Execute.Int 4;
    ] ;
  check bool "o1 equal" true (fbits o1 = fbits o1') ;
  check bool "o2 equal" true (fbits o2 = fbits o2')

(* Two store sites to the intermediate each get their own scope (review S2),
   and a consumer with helper functions is refused (review B2). *)
let test_store_sites_and_helpers () =
  let two_sites =
    {
      k_prod with
      kern_body =
        SSeq
          [
            store_at
              "dst"
              (EBinop (Mul, tid, EConst (CInt32 2l)))
              (EArrayRead ("src", tid));
            store_at
              "dst"
              (EBinop
                 (Add, EBinop (Mul, tid, EConst (CInt32 2l)), EConst (CInt32 1l)))
              (EArrayRead ("src", tid));
          ];
    }
  in
  let fused =
    Sarek_execute.Fusion_epilogue.fuse
      ~name:"two_sites"
      ~producer:two_sites
      ~p_array:"dst"
      ~consumer:k_cons
      ~shared:[("t", "dst")]
  in
  let rec blocks = function
    | SBlock (SLet (v, _, _)) when v.var_name = "ep_t" -> 1
    | SSeq ss -> List.fold_left (fun n s -> n + blocks s) 0 ss
    | SIf (_, t, f) -> blocks t + Option.fold ~none:0 ~some:blocks f
    | SLet (_, _, b)
    | SLetMut (_, _, b)
    | SBlock b
    | SPragma (_, b)
    | SWhile (_, b)
    | SFor (_, _, _, _, b) ->
        blocks b
    | _ -> 0
  in
  check int "two scoped store sites" 2 (blocks fused.kern_body) ;
  let with_helper =
    {
      k_cons with
      kern_funcs =
        [
          {
            hf_name = "f";
            hf_params = [ivar 30 "v"];
            hf_ret_type = TInt32;
            hf_body = SReturn (EVar (ivar 30 "v"));
          };
        ];
    }
  in
  match Sarek_execute.Fusion_epilogue.consumer with_helper with
  | Ok _ -> fail "a consumer with helper functions must be refused"
  | Error _ -> ()

(* Member id ranges are disjoint even when a member has large ids (a fused
   kernel), and the position parameters sit above them. *)
let test_horizontal_ids_disjoint () =
  let big =
    Sarek_execute.Fusion_epilogue.fuse
      ~name:"big"
      ~producer:k_prod
      ~p_array:"dst"
      ~consumer:k_cons
      ~shared:[("t", "dst")]
  in
  let fused = Sarek_execute.Fusion_horizontal.fuse ~name:"h" [big; k_prod] in
  let ids_of prefix =
    List.filter_map
      (function
        | DParam (v, _) when String.starts_with ~prefix v.var_name ->
            Some v.var_id
        | _ -> None)
      fused.kern_params
  in
  let m0 = ids_of "m0_" and m1 = ids_of "m1_" and hf = ids_of "hf" in
  check
    bool
    "members disjoint"
    true
    (List.for_all (fun i -> not (List.mem i m1)) m0) ;
  check
    bool
    "positions above members"
    true
    (List.for_all (fun h -> List.for_all (fun i -> h > i) (m0 @ m1)) hf)

let slice_unknown body =
  match
    Sarek_execute.Index_slice.run
      ~is_target:(fun a -> a = "dst")
      ~record:ignore
      ~scalars:[]
      ~dims:{Sarek_execute.Index_slice.block = (1, 1, 1); grid = (1, 1, 1)}
      body
  with
  | _ -> false
  | exception Sarek_execute.Index_slice.Unknown _ -> true

let test_slice_limits () =
  check
    bool
    "field store to the target is unknown"
    true
    (slice_unknown
       (SAssign
          (LRecordField (LArrayElem ("dst", tid), "f"), EConst (CFloat32 0.)))) ;
  let i = ivar 40 "i" in
  check
    bool
    "work budget fires"
    true
    (slice_unknown
       (SFor
          ( i,
            EConst (CInt32 0l),
            EConst (CInt32 Int32.max_int),
            Upto,
            store_at "dst" (EVar i) (EConst (CFloat32 0.)) )))

let () =
  run
    "launch window"
    [
      ( "deferral",
        [
          test_case "deferred until flush" `Quick test_deferred_until_flush;
          test_case "disabled is immediate" `Quick test_disabled_is_immediate;
          test_case "bounded window" `Quick test_window_is_bounded;
          test_case "domains" `Quick test_domains;
        ] );
      ( "grouping",
        [
          test_case "independent launches fused" `Quick test_independent_fused;
          test_case "dependent launches in order" `Quick test_dependent_in_order;
          test_case
            "reordered past independent"
            `Quick
            test_reordered_past_independent;
          test_case
            "different block not fused"
            `Quick
            test_different_block_not_fused;
        ] );
      ( "sync points",
        List.map
          (fun (what, marker, op) ->
            test_case what `Quick (fun () ->
                match sync_point what ~marker op () with
                | () -> ()
                | exception Failure m when what = "Kernel.launch (raw)" ->
                    failf "raw launch: %s" m))
          sync_points
        @ [
            test_case
              "upload of a bound buffer"
              `Quick
              test_upload_touching_flushes;
            test_case
              "upload of an unrelated buffer"
              `Quick
              test_upload_unrelated_keeps_pending;
          ] );
      ( "more sync points",
        List.map
          (fun (what, op) -> test_case what `Quick (issued_by what op))
          more_sync_points );
      ( "IR",
        [
          test_case "fusability verdicts" `Quick test_verdicts;
          test_case "parameter accesses" `Quick test_accesses;
          test_case
            "renaming cannot capture positions"
            `Quick
            test_rename_no_capture;
          test_case "grid limit" `Quick test_grid_limit;
        ] );
      ( "review round 3",
        [
          test_case "CPU device not deferred" `Quick test_cpu_device_not_deferred;
          test_case "shared-memory limit" `Quick test_shared_limit;
          test_case "overflow failure" `Quick test_overflow_failure;
          test_case
            "fused fallback, member fails"
            `Quick
            test_fused_fallback_member_fails;
          test_case
            "argument error at submit"
            `Quick
            test_argument_error_at_submit;
          test_case
            "sync waits for a flush in progress"
            `Quick
            test_sync_waits_for_flush_in_progress;
        ] );
      ( "epilogue",
        [
          test_case "index slice" `Quick test_index_slice;
          test_case "consumer shape" `Quick test_consumer_shape;
          test_case "merged when covered" `Quick test_epilogue_merged;
          test_case "not merged when uncovered" `Quick test_epilogue_uncovered;
          test_case "default backends" `Quick test_epilogue_off_by_default;
          test_case "failure falls back" `Quick test_epilogue_failure_falls_back;
          test_case "coverage per array" `Quick test_coverage_per_array;
          test_case "duplicate stores" `Quick test_duplicate_stores;
          test_case "chain on the interpreter" `Quick test_chain_on_interpreter;
          test_case
            "store sites and helpers"
            `Quick
            test_store_sites_and_helpers;
          test_case
            "horizontal ids disjoint"
            `Quick
            test_horizontal_ids_disjoint;
          test_case "slice limits" `Quick test_slice_limits;
        ] );
      ( "errors",
        [
          test_case
            "issue error re-raised after the rest"
            `Quick
            test_issue_error_reraised_after_rest;
          test_case
            "fused failure falls back"
            `Quick
            test_fused_failure_falls_back;
        ] );
    ]
