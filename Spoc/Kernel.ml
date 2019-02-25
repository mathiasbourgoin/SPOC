(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL - B license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify	and / or redistribute the software under the terms of the CeCILL - B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and / or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in - depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and / or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL - B license and that you accept its terms.
 *******************************************************************************)
type kernel

external relax_vector : ('a,'b) Vector.vector -> ('c,'d) Vector.vector = "%identity"

type ('a, 'b) kernelArgs =
  | VChar of ('a, 'b) Vector.vector (** unsigned char vector *)
  | VFloat32 of ('a, 'b) Vector.vector (** 32-bit float vector *)
  | VFloat64 of ('a, 'b) Vector.vector (** 64-bit float vector *)
  | VComplex32 of ('a, 'b) Vector.vector (** 32-bit complex vector *)
  | VInt32 of ('a, 'b) Vector.vector (** 32-bit int vector *)
  | VInt64 of ('a, 'b) Vector.vector (** 64-bit int vector *)
  | Int32 of int (** 32-bit int *)
  | Int64 of int (** 64-bit int *)
  | Float32 of float (** 32-bit float *)
  | Float64 of float (** 64-bit float *)
  | Custom of ('a,'b) Vector.custom
  | Vector of ('a, 'b) Vector.vector  (** generic vector type *)
  | VCustom of ('a,'b) Vector.vector  (** custom data type vector, see examples *)

type block =
  { mutable blockX : int; mutable blockY : int; mutable blockZ : int
  }

type grid = { mutable gridX : int; mutable gridY : int; mutable gridZ : int }

let nbCaches = 64

module Cuda =
struct
  type cuda_extra

  external cuda_create_extra : int -> cuda_extra = "spoc_cuda_create_extra"

  external cuda_compile :
    string ->
    string -> Devices.generalInfo -> kernel =
    "spoc_cuda_compile"

  external cuda_debug_compile :
    string ->
    string -> Devices.generalInfo -> kernel =
    "spoc_cuda_debug_compile" (* <- ATTENTION *)

  external cuda_load_param_vec :
    int ref -> cuda_extra -> Vector.device_vec -> ('a,'b) Vector.vector -> Devices.device -> unit =
    "spoc_cuda_load_param_vec_b" "spoc_cuda_load_param_vec_n"

  external cuda_custom_load_param_vec :
    int ref -> cuda_extra -> Vector.device_vec -> ('a,'b) Vector.vector -> unit =
    "spoc_cuda_custom_load_param_vec_b" "spoc_cuda_custom_load_param_vec_n"

  external cuda_load_param_int : int ref -> cuda_extra -> int -> unit =
    "spoc_cuda_load_param_int_b" "spoc_cuda_load_param_int_n"

  external cuda_load_param_int64 : int ref -> cuda_extra -> int -> unit =
    "spoc_cuda_load_param_int64_b" "spoc_cuda_load_param_int64_n"

  external cuda_load_param_float : int ref -> cuda_extra -> float -> unit =
    "spoc_cuda_load_param_float_b" "spoc_cuda_load_param_float_n"

  external cuda_load_param_float64 :
    int ref -> cuda_extra -> float -> unit =
    "spoc_cuda_load_param_float64_b" "spoc_cuda_load_param_float64_n"

  external set_block_shape : kernel -> block -> Devices.generalInfo -> unit =
    "spoc_cuda_set_block_shape"

  external cuda_launch_grid :
    int ref -> kernel -> grid -> block -> cuda_extra -> Devices.generalInfo -> int -> unit =
    "spoc_cuda_launch_grid_b" "spoc_cuda_launch_grid_n"

  external cuda_create_dummy_kernel : unit -> kernel =
    "spoc_cuda_create_dummy_kernel"

  let cudaKernelCache = ref [| |]

  let cuda_load cached id s1 s2 args =
    let path = Sys.argv.(0) in
    let kernelPath =
      (Filename.dirname path) ^ (Filename.dir_sep ^ (s1 ^ ".ptx"))
    in
    if cached
    then
      (if (Array.length !cudaKernelCache) = 0
       then
         cudaKernelCache :=
           Array.make (Devices.cuda_devices()) (Hashtbl.create nbCaches)
       ;
       (let hash = Digest.file kernelPath
        in
        try let ker = Hashtbl.find !cudaKernelCache.(id) hash in ker
        with
        | Not_found ->
          let src = ref "" in
          let ic = open_in kernelPath
          in
          ((try
              while true do src := !src ^ ((input_line ic) ^ "\n")
              done
            with | End_of_file -> ());
           close_in ic;
           let ker = cuda_compile !src s2
           in (Hashtbl.add !cudaKernelCache.(id) hash ker; ker))))
    else
      (let src = ref "" in
       let ic = open_in kernelPath
       in
       ((try while true do src := !src ^ ((input_line ic) ^ "\n") done
         with | End_of_file -> ());
        close_in ic;
        let ker = cuda_compile !src s2 in ker))

  let cuda_load_arg offset extra dev cuFun idx (arg: ('a,'b) kernelArgs) =
    let load_non_vect =
      function
      | Int32 i -> cuda_load_param_int offset extra i
      | Int64 i -> cuda_load_param_int64 offset extra i
      | Float32 f -> cuda_load_param_float offset extra f
      | Float64 f -> cuda_load_param_float64 offset extra f
      | _ -> failwith "CU LOAD ARG Type Not Implemented\n"

    and check_vect v =
      (if !Mem.auto
       then
         (
           (try Mem.to_device v dev;
              Devices.flush dev ()
            with
            | Cuda.ERROR_OUT_OF_MEMORY ->
              raise Cuda.ERROR_OUT_OF_MEMORY
           )
         )
       ;
       match arg with
       | VCustom v2 ->
         cuda_custom_load_param_vec offset extra
           (Vector.device_vec v2 `Cuda dev.Devices.general_info.Devices.id) v
       | _ ->
         cuda_load_param_vec offset extra
           (Vector.device_vec v `Cuda dev.Devices.general_info.Devices.id) v dev)
    in
    match arg with
    | VChar v | VFloat32 v
    | VComplex32 v
    | VInt32 v | VInt64 v
    | VFloat64 v -> check_vect v
    | VCustom (v : ('a, 'b) Vector.vector) -> check_vect v
    | _ -> load_non_vect arg
end

module OpenCL =
struct
  external opencl_compile : string -> string -> Devices.generalInfo -> kernel =
    "spoc_opencl_compile"

  external opencl_debug_compile : string -> string -> Devices.generalInfo -> kernel =
    "spoc_debug_opencl_compile"

  external opencl_load_param_vec :
    int ref -> kernel -> Vector.device_vec -> int -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_vec"

  external opencl_load_param_local_vec :
    int ref -> kernel -> int -> Vector.device_vec -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_local_vec"

  external opencl_load_param_int :
    int ref -> kernel -> int -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_int"

  external opencl_load_param_int64 :
    int ref -> kernel -> int -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_int64"

  external opencl_load_param_float :
    int ref -> kernel -> float -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_float"

  external opencl_load_param_float64 :
    int ref -> kernel -> float -> Devices.generalInfo -> unit =
    "spoc_opencl_load_param_float64"

  external opencl_launch_grid :
    kernel -> grid -> block -> Devices.generalInfo -> int -> unit =
    "spoc_opencl_launch_grid"

  external opencl_create_dummy_kernel : unit -> kernel =
    "spoc_opencl_create_dummy_kernel"

  let openCLKernelCache = ref [| |]

  let opencl_load cached debug id s1 s2 =
    let path = Sys.argv.(0) in
    let kernelPath =
      (Filename.dirname path) ^ (Filename.dir_sep ^ (s1 ^ ".cl"))
    in
    if cached
    then
      (if (Array.length !openCLKernelCache) = 0
       then
         openCLKernelCache :=
           Array.make (Devices.opencl_devices())
             (Hashtbl.create nbCaches)
       ;
       (let hash = Digest.file kernelPath
        in
        try let ker = Hashtbl.find !openCLKernelCache.(id) hash in ker
        with
        | Not_found ->
          let src = ref "" in
          let ic = open_in kernelPath
          in
          ((try
              while true do src := !src ^ ((input_line ic) ^ "\n")
              done
            with | End_of_file -> ());
           close_in ic;
           let ker = opencl_compile !src s2
           in (Hashtbl.add !openCLKernelCache.(id) hash ker; ker))))
    else
      (let src = ref "" in
       let ic = open_in kernelPath
       in
       ((try while true do src := !src ^ ((input_line ic) ^ "\n") done
         with | End_of_file -> ());
        close_in ic;
        let ker =
          if debug
          then opencl_debug_compile !src s2
          else opencl_compile !src s2
        in ker))

  let opencl_load_arg offset dev clFun idx (arg : ('a,'b) kernelArgs) =
    let load_non_vect =
      function
      | Int32 i -> opencl_load_param_int offset clFun i dev.Devices.general_info
      | Int64 i -> opencl_load_param_int64 offset clFun i dev.Devices.general_info
      | Float32 f ->
        opencl_load_param_float offset clFun f dev.Devices.general_info
      | Float64 f ->
        opencl_load_param_float64 offset clFun f dev.Devices.general_info
      | _ -> failwith "Cl LOAD ARG Type Not Implemented\n"

    and check_vect v =
      (if !Mem.auto
       then
         (if Vector.dev v <> (Vector.Dev dev) then Mem.to_device v dev ;
          Devices.flush dev ())
       ;
       opencl_load_param_vec offset clFun
         (Vector.device_vec v `OpenCL (dev.Devices.general_info.Devices.id - Devices.cuda_devices()))
         (Vector.get_vec_id v) dev.Devices.general_info)
    in
    match arg with
    | VChar v | VFloat32 v
    | VFloat64 v | VComplex32 v
    | VInt32 v | VInt64 v | VCustom v ->
      check_vect v
    | _ -> load_non_vect arg

end

exception ERROR_BLOCK_SIZE
exception ERROR_GRID_SIZE

let exec (args: ('a,'b) kernelArgs array) (block, grid) queue_id
    dev (bin: kernel) =
  match dev.Devices.specific_info with
  | Devices.CudaInfo cI ->
    let open Cuda in
    let cuFun = bin in
    let offset = ref 0 in
    let extra = cuda_create_extra (Array.length args)
    in
    (if
      (block.blockX > cI.Devices.maxThreadsDim.Devices.x) ||
      ((block.blockY > cI.Devices.maxThreadsDim.Devices.y) ||
       (block.blockZ > cI.Devices.maxThreadsDim.Devices.z))
     then raise ERROR_BLOCK_SIZE;
     if
       (grid.gridX > cI.Devices.maxGridSize.Devices.x) ||
       ((grid.gridY > cI.Devices.maxGridSize.Devices.y) ||
        (grid.gridZ > cI.Devices.maxGridSize.Devices.z))
     then
       (raise ERROR_GRID_SIZE)
    );
    Array.iteri (cuda_load_arg offset extra dev cuFun) (args: ('a,'b) kernelArgs array);
    (* set_block_shape cuFun block dev.general_info; *)
    cuda_launch_grid offset cuFun grid block extra dev.Devices.general_info queue_id
  | Devices.OpenCLInfo _ ->
    let open OpenCL in
    let clFun = bin in
    let offset = ref 0
    in
    (Array.iteri (opencl_load_arg offset dev clFun) (args: ('a,'b) kernelArgs array);
     opencl_launch_grid clFun grid block dev.Devices.general_info queue_id)
    
let compile_and_run (dev : Devices.device) ((block : block), (grid : grid))
    ?cached: (c = false) ?debug: (d = false) ?queue_id: (q = 0)
    ker =
  snd (fst ker) (block, grid) c d q dev

let max a b = if a < b then b else a

let load_source path file ext =
  let kernelPath =
(*    let path = Sys.argv.(0) in*)
    (*Filename.dirname path*) path ^ (Filename.dir_sep ^ (file ^ ext)) in
  let src = ref "" in
  let ic = open_in kernelPath
  in
  ((try while true do src := !src ^ ((input_line ic) ^ "\n") done
    with | End_of_file -> ());
   close_in ic;
   !src)

exception No_source_for_device of Devices.device
exception Not_compiled_for_device of Devices.device

class virtual ['a, 'b] spoc_kernel file (func: string) =
  object (self)
    val file_file = file
    val kernel_name = func
    val mutable source_path = Filename.dirname Sys.argv.(0)
    val mutable cuda_sources =
      try [ load_source (Filename.dirname Sys.argv.(0)) file ".ptx" ] with | _ -> []

    val mutable opencl_sources =
      try [ load_source (Filename.dirname Sys.argv.(0)) file ".cl" ] with | _ -> []

    val binaries = Hashtbl.create 8

    method get_binaries () = binaries
    method reset_binaries () = Hashtbl.clear binaries

    method set_source_path path =
      source_path <- path

    method get_cuda_sources () = cuda_sources
    method set_cuda_sources s =
      cuda_sources <- [s]

    method get_opencl_sources () = opencl_sources
    method set_opencl_sources s =
      opencl_sources <- [s]

    method reload_sources () =
      cuda_sources <-
        (try [ load_source source_path file ".ptx" ] with | _ -> []);
      opencl_sources <-
        (try [ load_source source_path file ".cl" ] with | _ -> [])

    method compile ?debug: (d = false) =
      fun dev ->
        try Hashtbl.find binaries dev
        with
        | Not_found ->
          let bin =
            (match dev.Devices.specific_info with
             | Devices.CudaInfo _ ->
               let open Cuda in
               begin
                 match cuda_sources with
                 | [] -> raise (No_source_for_device dev)
                 | t:: q -> if d then
                     cuda_debug_compile t kernel_name dev.Devices.general_info
                   else
                     cuda_compile t kernel_name dev.Devices.general_info
               end
             | Devices.OpenCLInfo _ ->
               let open OpenCL in
               begin
                 match opencl_sources with
                 | [] -> raise (No_source_for_device dev)
                 | t:: q -> if d then
                     opencl_debug_compile t kernel_name dev.Devices.general_info
                   else
                     opencl_compile t kernel_name dev.Devices.general_info
               end
            )
          in (Hashtbl.add binaries dev bin); bin

    method virtual exec : 'a -> (block * grid) ->
      int -> Devices.device -> kernel -> unit

    method virtual list_to_args : 'b -> 'a
    method virtual args_to_list : 'a -> 'b

    method run (args:'a) ((block: block), (grid: grid)) (queue_id: int) (dev: Devices.device) =
      let bin =
        try Hashtbl.find binaries dev with
        | Not_found ->
          (
            try self#compile ~debug: true dev with
            |e -> raise e
            (*| _ -> raise (Not_compiled_for_device dev)*)
          );
          Hashtbl.find binaries dev
      in
        self#exec args (block, grid) queue_id dev bin

    method compile_and_run (args:'a) ((block: block), (grid: grid))
        ?debug: (d = false) (queue_id: int) (dev: Devices.device) =
      let bin =
        self#compile ~debug: d dev;
        Hashtbl.find binaries dev
      in
        self#exec args (block, grid) queue_id dev bin
  end

let run (dev: Devices.device) ((block: block), (grid: grid)) (k: ('a, 'b) spoc_kernel) (args:'a) = k#run args (block, grid) 0 dev

let compile (dev: Devices.device) (k: ('a, 'b) spoc_kernel) = ignore(k#compile ~debug: false dev)

let set_arg env i arg =
  env.(i) <-
    (match Vector.kind arg with
     | Vector.Float32 x -> VFloat32 arg
     | Vector.Char x -> VChar arg
     |	Vector.Float64 x -> VFloat64 arg
     | Vector.Int32 x -> VInt32 arg
     | Vector.Int64 x -> VInt64 arg
     | Vector.Complex32 x ->
       VComplex32 arg
     | _ -> assert false
    )
