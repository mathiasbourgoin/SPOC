(** V2 transform helpers (temporarily stubbed).

    The legacy SPOC-based implementation has been removed; these signatures
    describe the intended V2 surface (Sarek_core vectors/devices, V2 IR). All
    functions currently raise [Failure "unimplemented"] until the port is
    complete. *)

open Kirc_Ast

(** Convert a scalar kernel argument into a vector argument node. *)
val a_to_vect : k_ext -> k_ext

(** Rewrite a return expression into a vector write at the given index. *)
val a_to_return_vect : k_ext -> k_ext -> k_ext -> k_ext

(** Translate a V2 vector into the argument list expected by [Execute]. *)
val arg_of_vec : ('a, 'b) Sarek_core.Vector.t -> Execute.vector_arg list

(** Launch a compiled kernel against the V2 runtime. *)
val launch_kernel_with_args :
  ir:Sarek_ir.kernel ->
  device:Sarek_core.Device.t ->
  grid:Sarek_core.Runtime.dims ->
  block:Sarek_core.Runtime.dims ->
  args:Execute.vector_arg list ->
  unit

(** Heuristic 1D grid/block computation based on device limits and input size.
*)
val compute_grid_block_1D :
  device:Sarek_core.Device.t ->
  ('a, 'b) Sarek_core.Vector.t ->
  Sarek_core.Runtime.dims * Sarek_core.Runtime.dims

(** Apply a transformer function to a kernel AST node. *)
val propagate : (k_ext -> k_ext) -> k_ext -> k_ext

(** Parallel zip-with using two input vectors. *)
val map2 :
  ('a, 'b, 'c -> 'd -> 'e, 'f, 'g) Kirc_types.sarek_kernel ->
  ?device:Sarek_core.Device.t ->
  ('c, 'i) Sarek_core.Vector.t ->
  ('d, 'k) Sarek_core.Vector.t ->
  ('e, 'm) Sarek_core.Vector.t

(** Reduction over a single input vector. *)
val reduce :
  ('a, 'b, 'c -> 'c -> 'd, 'e, 'f) Kirc_types.sarek_kernel ->
  ?device:Sarek_core.Device.t ->
  ('c, 'i) Sarek_core.Vector.t ->
  'd

(** Rebuild a kernel pair with a new AST and ML implementation. *)
val build_new_ker :
  'spoc ->
  ('a, 'b, 'c) Kirc_types.kirc_kernel ->
  k_ext ->
  'ml_fun ->
  'spoc * ('a, 'b, 'c) Kirc_types.kirc_kernel

(** Map over a single vector. *)
val map :
  ('a, 'b, 'c, 'd, 'e) Kirc_types.sarek_kernel ->
  ?device:Sarek_core.Device.t ->
  ('d, 'h) Sarek_core.Vector.t ->
  ('f, 'g) Sarek_core.Vector.t

exception Zip of string

(** Zip two vectors together with a kernel-defined function. *)
val zip :
  ('a, 'b, 'c, 'd, 'e) Kirc_types.sarek_kernel ->
  ?device:Sarek_core.Device.t ->
  ('f, 'g) Sarek_core.Vector.t ->
  ('h, 'i) Sarek_core.Vector.t ->
  ('j, 'k) Sarek_core.Vector.t

(** Detect intermediate arrays between two kernel bodies. *)
val detect_intermediates : k_ext -> k_ext -> string list

(** Attempt to fuse two kernel bodies, returning the fused body if possible. *)
val try_fuse_bodies : k_ext -> k_ext -> k_ext option

(** Fuse a pipeline of kernel bodies, returning the fused body and eliminated
    intermediate arrays. *)
val fuse_pipeline_bodies : k_ext list -> k_ext * string list

(** Pipelined fusion helpers. *)
module Pipeline : sig
  type 'a stage = {body : k_ext; info : 'a}

  val of_bodies : k_ext list -> unit stage list

  val optimize : 'a stage list -> k_ext * string list

  val can_fuse_stages : 'a stage -> 'b stage -> bool
end
