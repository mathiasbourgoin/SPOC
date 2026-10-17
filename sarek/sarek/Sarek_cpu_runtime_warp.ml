(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Warp shuffles for the Native backend: the slow, sequential, correct version.

    A warp is simulated: [Sarek_warp.simulated_width ()] consecutive threads of
    the block (linear thread id, x fastest). A shuffle is one exchange through a
    per-block typed array held in the block's shared memory:

    + each thread publishes its value at its linear thread id;
    + block barrier (every thread of the block reaches the call, which the
      convergence rule guarantees);
    + each thread reads the slot of its source lane, from
      {!Sarek_warp.source_lane};
    + a second barrier, so a following shuffle cannot overwrite a slot that a
      slower thread has not read yet.

    The exchange arrays are typed ([int32 array], [float array]); there is no
    [Obj]. The cost is two barriers per call and one slot per thread of the
    block: this is the correctness oracle, not a fast path. *)

open Sarek_cpu_runtime_types

(** The four shuffle flavours, re-exported so generated kernels name them
    without depending on the IR library. *)
type op = Sarek_warp.op = Idx | Down | Up | Xor

let linear_thread_id (st : thread_state) =
  let tx = Int32.to_int st.thread_idx_x
  and ty = Int32.to_int st.thread_idx_y
  and tz = Int32.to_int st.thread_idx_z
  and bx = Int32.to_int st.block_dim_x
  and by = Int32.to_int st.block_dim_y in
  tx + (bx * (ty + (by * tz)))

let block_size (st : thread_state) =
  Int32.to_int st.block_dim_x
  * Int32.to_int st.block_dim_y
  * Int32.to_int st.block_dim_z

(** The simulated warp width as a launch-time quantity. *)
let width () = Sarek_warp.simulated_width ()

(** [warp_size] as seen by a kernel. *)
let warp_size () = Int32.of_int (width ())

(** [lane_id] as seen by a kernel: linear thread id modulo the warp width. *)
let lane_id (st : thread_state) =
  Int32.of_int
    (Sarek_warp.lane_of_thread ~width:(width ()) (linear_thread_id st))

(* The source slot of this thread, refusing a block that is not a whole number of
   warps: its last warp would read slots that do not exist. The launch rule
   (Execute) refuses such a launch before it gets here, so this is the
   defensive half of the same invariant, not a second policy. *)
let source_slot st op x =
  let w = width () in
  let n = block_size st in
  if n mod w <> 0 then
    invalid_arg
      (Printf.sprintf
         "warp shuffle: block of %d threads is not a multiple of the warp \
          width %d"
         n
         w) ;
  let t = linear_thread_id st in
  let lane = Sarek_warp.lane_of_thread ~width:w t in
  t - lane + Sarek_warp.source_lane op ~width:w ~lane (Int32.to_int x)

let exchange ~(slots : 'a array) st ~src v =
  slots.(linear_thread_id st) <- v ;
  st.barrier () ;
  let r = slots.(src) in
  st.barrier () ;
  r

(** Int32 shuffle. [op] selects idx, down, up or xor. *)
let shuffle_i32 st shared op v x =
  let src = source_slot st op x in
  let slots = alloc_shared_int32 shared "__warp_xchg_i32" (block_size st) 0l in
  exchange ~slots st ~src v

(** Float32 shuffle. float32 values are carried as OCaml floats on the CPU. *)
let shuffle_f32 st shared op v x =
  let src = source_slot st op x in
  let slots = alloc_shared_float shared "__warp_xchg_f" (block_size st) 0. in
  exchange ~slots st ~src v

(** Float64 shuffle. *)
let shuffle_f64 st shared op v x =
  let src = source_slot st op x in
  let slots = alloc_shared_float shared "__warp_xchg_f64" (block_size st) 0. in
  exchange ~slots st ~src v
