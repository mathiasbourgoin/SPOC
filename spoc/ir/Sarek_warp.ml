(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Sarek_warp - the one definition of warp shuffle semantics.

    The four shuffle primitives ([warp_shuffle], [warp_shuffle_down],
    [warp_shuffle_up], [warp_shuffle_xor]) and the two warp introspection
    constants ([warp_size], [lane_id]) reach the IR as [EIntrinsic] nodes. The
    shuffles are TYPED in the IR: the front end appends the value type to the
    name ([warp_shuffle_down_f32]) so a backend never has to infer the type of
    an expression to choose a spelling.

    Contract (docs/plans/warp-shuffle-intake-2026-10-17.md, section 4.3), for
    lane [i] of a warp of width [W] (a power of two) and operand [x] read as a
    two's-complement int32:

    - [warp_shuffle v x]: source lane [x land (W-1)];
    - [warp_shuffle_down v d]: [i + d], or [i] when [d < 0] or [i + d >= W];
    - [warp_shuffle_up v d]: [i - d], or [i] when [d < 0] or [i - d < 0];
    - [warp_shuffle_xor v m]: [i lxor m], or [i] when [m < 0] or
      [i lxor m >= W].

    {!source_lane} is that table, executable. Native and the interpreter use it
    directly; the source backends emit equivalent guards, and the parity tests
    compare every backend with it. *)

type op = Idx | Down | Up | Xor

type value_type = I32 | F32 | F64

type call = Shuffle of op * value_type | Warp_size | Lane_id

let op_base = function
  | Idx -> "warp_shuffle"
  | Down -> "warp_shuffle_down"
  | Up -> "warp_shuffle_up"
  | Xor -> "warp_shuffle_xor"

let type_suffix = function I32 -> "i32" | F32 -> "f32" | F64 -> "f64"

let all_ops = [Idx; Down; Up; Xor]

let all_types = [I32; F32; F64]

(** The IR name of a typed shuffle, e.g. ["warp_shuffle_down_f32"]. *)
let ir_name op ty = op_base op ^ "_" ^ type_suffix ty

let typed_names =
  List.concat_map
    (fun op -> List.map (fun ty -> (ir_name op ty, Shuffle (op, ty))) all_types)
    all_ops

(** Classify an IR intrinsic name. [None] for anything that is not a warp
    shuffle or a warp introspection constant. *)
let classify name =
  match name with
  | "warp_size" -> Some Warp_size
  | "lane_id" -> Some Lane_id
  | _ -> List.assoc_opt name typed_names

(** The user-facing names of the four shuffles (before typing). *)
let user_shuffle_names = List.map op_base all_ops

(** Is [name] one of the four user-facing shuffle names? *)
let is_user_shuffle_name name = List.mem name user_shuffle_names

(** The op of a user-facing shuffle name. *)
let op_of_user_name name = List.find_opt (fun op -> op_base op = name) all_ops

(** Does this IR intrinsic name denote a shuffle (a warp collective)? *)
let is_shuffle_name name =
  match classify name with Some (Shuffle _) -> true | _ -> false

(** Does this IR intrinsic name depend on the warp structure (shuffle,
    [warp_size], [lane_id])? Passes that change which threads run, or the block
    shape, must leave such a kernel alone. *)
let is_warp_name name = classify name <> None

(** Index of the lane read by lane [lane] for operand [x] in a warp of [width]
    lanes. The result is always in [0, width). [x] is the int32 operand
    sign-extended to an OCaml int. *)
let source_lane op ~width ~lane x =
  match op with
  | Idx -> x land (width - 1)
  | Down -> if x < 0 || x >= width - lane then lane else lane + x
  | Up -> if x < 0 || lane - x < 0 then lane else lane - x
  | Xor ->
      if x < 0 then lane
      else
        let s = lane lxor x in
        if s >= width then lane else s

(** {1 Simulated warp width}

    Native and the interpreter have no hardware warp. They simulate one of
    [simulated_width ()] lanes (default 32, the width of the CUDA and Apple
    devices and of RDNA). The width is a launch-time parameter in the sense that
    it is read at each launch; {!with_width} is the test hook that lets the
    parity tests run the simulation at a GPU's width. *)

let default_width = 32

let is_valid_width w = w >= 1 && w land (w - 1) = 0 && w <= 1024

let simulated = Atomic.make default_width

let simulated_width () = Atomic.get simulated

(** Run [f] with the simulated width set to [w], restoring the previous width
    afterwards. Raises [Invalid_argument] when [w] is not a power of two in
    [1, 1024]. Not re-entrant across domains: it is a test hook. *)
let with_width w f =
  if not (is_valid_width w) then
    invalid_arg
      (Printf.sprintf "Sarek_warp.with_width: %d is not a power of two" w) ;
  let previous = Atomic.exchange simulated w in
  Fun.protect ~finally:(fun () -> Atomic.set simulated previous) f

(** Linear thread id of [(tx, ty, tz)] in a block of [(bx, by, _)] (x fastest).
*)
let linear_thread_id ~block:(bx, by, _) (tx, ty, tz) =
  tx + (bx * (ty + (by * tz)))

(** Lane of a linear thread id in a warp of [width] lanes. *)
let lane_of_thread ~width t = t land (width - 1)

(** Is a block of [bx * by * bz] threads a whole number of warps of [width]? *)
let block_is_whole_warps ~block:(bx, by, bz) ~width = bx * by * bz mod width = 0
