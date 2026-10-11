(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* One launch on every process-wide CPU worker pool, and the measurements the
   idle-pool test and benchmark take around it. *)

open Sarek_ir_types

let block = (8, 1, 1)

let grid = (4, 1, 1)

let elements = 32

(* dst.(idx) <- idx + 1, built by hand so the interpreter launch needs no PPX. *)
let interp_ir () : kernel =
  let make_var id name ty =
    {var_name = name; var_id = id; var_type = ty; var_mutable = false}
  in
  let dst = make_var 0 "dst" (TVec TInt32) in
  let idx = make_var 1 "idx" TInt32 in
  let body =
    SLet
      ( idx,
        EIntrinsic ([], "global_thread_id", []),
        SAssign
          ( LArrayElem ("dst", EVar idx),
            EBinop (Add, EVar idx, EConst (CInt32 1l)) ) )
  in
  {
    default_kernel with
    kern_name = "cpu_pool_workload";
    kern_params =
      [DParam (dst, Some {arr_elttype = TInt32; arr_memspace = Global})];
    kern_body = body;
  }

let increment state _shared (arr : int array) =
  let i = Int32.to_int (Sarek.Sarek_cpu_runtime.global_idx_x state) in
  arr.(i) <- arr.(i) + 1

(* The parallel interpreter, the Native barrier and barrier-free pools and a
   fission queue, once each. Not [Sarek_cpu_runtime.run_parallel]: with it, a
   32-core host would need 129 domains, over OCaml's limit of 128. *)
let touch_every_pool () =
  let dst = Array.make elements (Sarek.Sarek_ir_interp.VInt32 0l) in
  Sarek.Sarek_ir_interp.run_kernel
    (interp_ir ())
    ~block
    ~grid
    [("dst", Sarek.Sarek_ir_interp.ArgArray dst)] ;
  let arr = Array.make elements 0 in
  Sarek.Sarek_cpu_runtime.run_threadpool
    ~has_barriers:false
    ~block
    ~grid
    increment
    arr ;
  Sarek.Sarek_cpu_runtime.run_threadpool
    ~has_barriers:true
    ~block
    ~grid
    increment
    arr ;
  Sarek.Sarek_cpu_runtime.enqueue_fission
    ~queue_id:0
    ~kernel:(fun ~mode:_ ~block:_ ~grid:_ _ -> ())
    ~args:[||]
    ~block
    ~grid
    () ;
  Sarek.Sarek_cpu_runtime.flush_fission ()

(* OS threads of this process (each domain is one, plus its backup thread), or
   [None] where /proc is unavailable. *)
let os_threads () =
  match In_channel.with_open_text "/proc/self/status" In_channel.input_all with
  | exception Sys_error _ -> None
  | status ->
      String.split_on_char '\n' status
      |> List.find_map (fun line ->
          match String.split_on_char ':' line with
          | ["Threads"; n] -> int_of_string_opt (String.trim n)
          | _ -> None)

(* Mean microseconds per minor collection over an allocating loop of
   [iterations] small blocks, and the number of collections. *)
let minor_gc_cost ~iterations =
  let c0 = (Gc.quick_stat ()).Gc.minor_collections in
  let t0 = Unix.gettimeofday () in
  for i = 1 to iterations do
    ignore (Sys.opaque_identity (Array.make 8 i))
  done ;
  let elapsed = Unix.gettimeofday () -. t0 in
  let collections = (Gc.quick_stat ()).Gc.minor_collections - c0 in
  (elapsed *. 1e6 /. float_of_int (max 1 collections), collections)
