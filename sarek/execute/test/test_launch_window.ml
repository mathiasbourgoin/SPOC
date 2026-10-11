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
    then failwith ("probe launch refused: " ^ kernel_name) ;
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

let launches () =
  List.filter
    (fun e -> String.starts_with ~prefix:"launch" e)
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
    ["launch sarek_hfused_0 gx=10 nargs=18"]
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
    ["launch k_a gx=1 nargs=4"; "launch sarek_hfused_0 gx=2 nargs=12"]
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
    ["launch sarek_hfused_0 gx=2 nargs=12"]
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
