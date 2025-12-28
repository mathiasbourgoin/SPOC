(******************************************************************************
 * Sarek GPU Standard Library
 *
 * Provides GPU thread/block indices, dimensions, and synchronization primitives.
 * Uses %sarek_intrinsic to define GPU intrinsics that auto-register.
 *
 * Each intrinsic generates:
 * - func_device : device -> string  (the device code generator)
 * - func_device_ref : (device -> string) ref  (for extension chaining)
 * - func : host-side OCaml implementation (dummy for GPU-only values)
 * - Registry entry for JIT code generation
 ******************************************************************************)

let dev cuda opencl d = Sarek.Sarek_registry.cuda_or_opencl d cuda opencl

(******************************************************************************
 * Thread indices within the block
 ******************************************************************************)

let%sarek_intrinsic (thread_idx_x : int32) =
  {device = dev "threadIdx.x" "get_local_id(0)"; ocaml = 0l}

let%sarek_intrinsic (thread_idx_y : int32) =
  {device = dev "threadIdx.y" "get_local_id(1)"; ocaml = 0l}

let%sarek_intrinsic (thread_idx_z : int32) =
  {device = dev "threadIdx.z" "get_local_id(2)"; ocaml = 0l}

(******************************************************************************
 * Block indices within the grid
 ******************************************************************************)

let%sarek_intrinsic (block_idx_x : int32) =
  {device = dev "blockIdx.x" "get_group_id(0)"; ocaml = 0l}

let%sarek_intrinsic (block_idx_y : int32) =
  {device = dev "blockIdx.y" "get_group_id(1)"; ocaml = 0l}

let%sarek_intrinsic (block_idx_z : int32) =
  {device = dev "blockIdx.z" "get_group_id(2)"; ocaml = 0l}

(******************************************************************************
 * Block dimensions (number of threads per block)
 ******************************************************************************)

let%sarek_intrinsic (block_dim_x : int32) =
  {device = dev "blockDim.x" "get_local_size(0)"; ocaml = 0l}

let%sarek_intrinsic (block_dim_y : int32) =
  {device = dev "blockDim.y" "get_local_size(1)"; ocaml = 0l}

let%sarek_intrinsic (block_dim_z : int32) =
  {device = dev "blockDim.z" "get_local_size(2)"; ocaml = 0l}

(******************************************************************************
 * Grid dimensions (number of blocks)
 ******************************************************************************)

let%sarek_intrinsic (grid_dim_x : int32) =
  {device = dev "gridDim.x" "get_num_groups(0)"; ocaml = 0l}

let%sarek_intrinsic (grid_dim_y : int32) =
  {device = dev "gridDim.y" "get_num_groups(1)"; ocaml = 0l}

let%sarek_intrinsic (grid_dim_z : int32) =
  {device = dev "gridDim.z" "get_num_groups(2)"; ocaml = 0l}

(******************************************************************************
 * Convenience: global thread ID
 ******************************************************************************)

let%sarek_intrinsic (global_thread_id : int32) =
  {
    device = dev "(threadIdx.x + blockIdx.x * blockDim.x)" "get_global_id(0)";
    ocaml = 0l;
  }

(******************************************************************************
 * Synchronization
 ******************************************************************************)

let%sarek_intrinsic (block_barrier : unit -> unit) =
  {
    (* Use %s placeholder to consume the Unit argument (which becomes "").
       This prevents the code generator from adding extra parentheses.
       Include semicolon for proper statement termination when barrier
       is the last statement before closing brace. *)
    device = dev "__syncthreads();%s" "barrier(CLK_LOCAL_MEM_FENCE);%s";
    ocaml = (fun () -> ());
  }

let%sarek_intrinsic (return_unit : unit -> unit) =
  {device = (fun _ -> "return"); ocaml = (fun () -> ())}

(******************************************************************************
 * Atomic Operations
 ******************************************************************************)

(* Atomic add: atomically adds value to the memory location and returns old value.
   For GPU: uses atomicAdd (CUDA) or atomic_add (OpenCL).
   For CPU: uses Atomic.fetch_and_add which is correct for parallel execution. *)
let%sarek_intrinsic (atomic_add_int32 : int32 array -> int32 -> int32 -> int32)
    =
  {
    (* Template: atomic_add(arr + idx, val)
       CUDA: atomicAdd returns old value
       OpenCL: atomic_add returns old value *)
    device = dev "atomicAdd(%s + %s, %s)" "atomic_add(%s + %s, %s)";
    ocaml =
      (fun arr idx value ->
        (* For OCaml, we just do non-atomic add since interpreter is sequential *)
        let i = Stdlib.Int32.to_int idx in
        let old = arr.(i) in
        arr.(i) <- Stdlib.Int32.add old value ;
        old);
  }

(* Atomic increment: atomically increments memory location, returns old value *)
let%sarek_intrinsic (atomic_inc_int32 : int32 array -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, 1)" "atomic_inc(%s + %s)";
    ocaml =
      (fun arr idx ->
        let i = Stdlib.Int32.to_int idx in
        let old = arr.(i) in
        arr.(i) <- Stdlib.Int32.add old 1l ;
        old);
  }

(* Global memory atomic add - uses Vector type for global memory *)
let%sarek_intrinsic
    (atomic_add_global_int32 : int32 vector -> int32 -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, %s)" "atomic_add(%s + %s, %s)";
    ocaml =
      (fun vec idx value ->
        let i = Stdlib.Int32.to_int idx in
        let old = Spoc.Mem.get vec i in
        Spoc.Mem.set vec i (Stdlib.Int32.add old value) ;
        old);
  }

(* Global memory atomic increment *)
let%sarek_intrinsic (atomic_inc_global_int32 : int32 vector -> int32 -> int32) =
  {
    device = dev "atomicAdd(%s + %s, 1)" "atomic_inc(%s + %s)";
    ocaml =
      (fun vec idx ->
        let i = Stdlib.Int32.to_int idx in
        let old = Spoc.Mem.get vec i in
        Spoc.Mem.set vec i (Stdlib.Int32.add old 1l) ;
        old);
  }

(******************************************************************************
 * Type conversions
 ******************************************************************************)

let%sarek_intrinsic (float_of_int : int32 -> float) =
  {
    device = (fun _ -> "(float)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

(* Alias for float_of_int - OCaml convention *)
let%sarek_intrinsic (float : int32 -> float32) =
  {
    device = (fun _ -> "(float)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

let%sarek_intrinsic (float64_of_int : int32 -> float) =
  {
    device = (fun _ -> "(double)");
    ocaml = (fun i -> Stdlib.float_of_int (Stdlib.Int32.to_int i));
  }

let%sarek_intrinsic (int_of_float : float -> int32) =
  {
    device = (fun _ -> "(int)");
    ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
  }

let%sarek_intrinsic (int_of_float64 : float -> int32) =
  {
    device = (fun _ -> "(int)");
    ocaml = (fun f -> Stdlib.Int32.of_int (Stdlib.int_of_float f));
  }

(******************************************************************************
 * Integer power
 ******************************************************************************)

let%sarek_intrinsic (spoc_powint : int32 -> int32 -> int32) =
  {
    device = (fun _ -> "spoc_powint");
    ocaml =
      (fun base exp ->
        let rec pow b e acc =
          if e = 0l then acc
          else pow b (Stdlib.Int32.sub e 1l) (Stdlib.Int32.mul acc b)
        in
        pow base exp 1l);
  }
