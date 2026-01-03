open Sarek_core
open Kirc_Ast

(* NOTE: This module is being rewritten for the V2 runtime. The APIs stay
   exported, but every function currently raises to make legacy usage fail
   fast. *)

module K_types = Kirc_types
module Sarek_ir = Sarek_ir
module Sarek_fusion = Sarek_fusion

(************** Composition ****************)

let a_to_vect (_ : k_ext) : k_ext =
  failwith "Transform.a_to_vect: unimplemented in V2-only placeholder"

let a_to_return_vect (_k1 : k_ext) (_k2 : k_ext) (_idx : k_ext) : k_ext =
  failwith "Transform.a_to_return_vect: unimplemented in V2-only placeholder"

let arg_of_vec (_ : ('a, 'b) Vector.t) : Execute.vector_arg list =
  failwith "Transform.arg_of_vec: unimplemented in V2-only placeholder"

let launch_kernel_with_args ~(ir : Sarek_ir.kernel) ~(device : Device.t)
    ~(grid : Runtime.dims) ~(block : Runtime.dims)
    ~(args : Execute.vector_arg list) : unit =
  ignore (ir, device, grid, block, args) ;
  failwith
    "Transform.launch_kernel_with_args: unimplemented in V2-only placeholder"

let compute_grid_block_1D ~(device : Device.t) (vec_in : ('a, 'b) Vector.t) :
    Runtime.dims * Runtime.dims =
  ignore (device, vec_in) ;
  failwith
    "Transform.compute_grid_block_1D: unimplemented in V2-only placeholder"

let propagate (_f : k_ext -> k_ext) (_expr : k_ext) : k_ext =
  failwith "Transform.propagate: unimplemented in V2-only placeholder"

let map2 (_ker : ('a, 'b, 'c -> 'd -> 'e, 'f, 'g) K_types.sarek_kernel)
    ?device:(_ : Device.t option) (_vec_in1 : ('c, 'i) Vector.t)
    (_vec_in2 : ('d, 'k) Vector.t) : ('e, 'm) Vector.t =
  failwith "Transform.map2: unimplemented in V2-only placeholder"

let reduce (_ker : ('a, 'b, 'c -> 'c -> 'd, 'e, 'f) K_types.sarek_kernel)
    ?device:(_ : Device.t option) (_vec_in1 : ('c, 'i) Vector.t) : 'd =
  failwith "Transform.reduce: unimplemented in V2-only placeholder"

let build_new_ker (_spoc_ker : 'spoc)
    (_kir_ker : ('a, 'b, 'c) K_types.kirc_kernel) (_ker : k_ext)
    (_ml_fun : 'ml_fun) : 'spoc * ('a, 'b, 'c) K_types.kirc_kernel =
  failwith "Transform.build_new_ker: unimplemented in V2-only placeholder"

let map (_f : ('a, 'b, 'c, 'd, 'e) K_types.sarek_kernel)
    ?device:(_ : Device.t option) (_vec_in : ('d, 'h) Vector.t) :
    ('f, 'g) Vector.t =
  failwith "Transform.map: unimplemented in V2-only placeholder"

exception Zip of string

let zip (_f : ('a, 'b, 'c, 'd, 'e) K_types.sarek_kernel)
    ?device:(_ : Device.t option) (_vec_in1 : ('f, 'g) Vector.t)
    (_vec_in2 : ('h, 'i) Vector.t) : ('j, 'k) Vector.t =
  failwith "Transform.zip: unimplemented in V2-only placeholder"

(** {1 Kernel Fusion}

    Automatic fusion of producer-consumer kernels to eliminate intermediate
    arrays and reduce memory traffic. *)

let detect_intermediates (_producer_body : k_ext) (_consumer_body : k_ext) :
    string list =
  failwith
    "Transform.detect_intermediates: unimplemented in V2-only placeholder"

let try_fuse_bodies (_producer_body : k_ext) (_consumer_body : k_ext) :
    k_ext option =
  failwith "Transform.try_fuse_bodies: unimplemented in V2-only placeholder"

let fuse_pipeline_bodies (_bodies : k_ext list) : k_ext * string list =
  failwith
    "Transform.fuse_pipeline_bodies: unimplemented in V2-only placeholder"

module Pipeline = struct
  type 'a stage = {body : k_ext; info : 'a}

  let of_bodies (_bodies : k_ext list) : unit stage list =
    failwith
      "Transform.Pipeline.of_bodies: unimplemented in V2-only placeholder"

  let optimize (_stages : 'a stage list) : k_ext * string list =
    failwith "Transform.Pipeline.optimize: unimplemented in V2-only placeholder"

  let can_fuse_stages (_s1 : 'a stage) (_s2 : 'b stage) : bool =
    failwith
      "Transform.Pipeline.can_fuse_stages: unimplemented in V2-only placeholder"
end
