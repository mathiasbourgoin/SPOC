(******************************************************************************
 * Kirc_v2 - Phase 4 Kernel Type
 *
 * Defines a new kernel record type with lazy IR generation for the unified
 * execution architecture. Coexists with the existing Kirc module during
 * the transition period.
 *
 * Key differences from Kirc:
 * - Lazy IR: Sarek_ir.kernel is generated only when needed (JIT backends)
 * - Native function: Pre-compiled OCaml function for Direct backends
 * - Typed AST: Access to the typed AST for Custom backends
 * - Clean separation: No legacy Kirc_Ast.k_ext in the primary interface
 ******************************************************************************)

open Sarek_framework
open Sarek_core

(** {1 Kernel Type} *)

(** Phase 4 kernel record with lazy IR generation *)
type 'a kernel_v2 = {
  name : string;  (** Kernel name (used for compilation cache key) *)
  ir : Sarek_ir.kernel Lazy.t;
      (** Lazy IR generation - only forced for JIT backends *)
  native_fn :
    (block:Framework_sig.dims -> grid:Framework_sig.dims -> Obj.t array -> unit)
    option;
      (** Pre-compiled OCaml function for Direct backends *)
  param_types : Sarek_ir.elttype list;
      (** Parameter types for argument marshalling *)
  extensions : Kirc.extension array;
      (** Extensions required (ExFloat32, ExFloat64) *)
}

(** {1 Constructors} *)

(** Create a kernel with both IR and native function *)
let make ~name ~ir ~native_fn ~param_types ?(extensions = [||]) () =
  {name; ir; native_fn; param_types; extensions}

(** Create a JIT-only kernel (no native function) *)
let make_jit ~name ~ir ~param_types ?(extensions = [||]) () =
  {name; ir; native_fn = None; param_types; extensions}

(** Create a Native-only kernel (no IR) *)
let make_native ~name ~native_fn ~param_types () =
  {
    name;
    ir = lazy (failwith "Native-only kernel has no IR");
    native_fn = Some native_fn;
    param_types;
    extensions = [||];
  }

(** {1 Accessors} *)

(** Get the kernel name *)
let name k = k.name

(** Get the IR (forces evaluation if lazy) *)
let ir k = Lazy.force k.ir

(** Check if native function is available *)
let has_native k = Option.is_some k.native_fn

(** Get the native function (raises if not available) *)
let native_fn k =
  match k.native_fn with
  | Some fn -> fn
  | None -> failwith ("Kernel " ^ k.name ^ " has no native function")

(** Get parameter types *)
let param_types k = k.param_types

(** Get extensions *)
let extensions k = k.extensions

(** {1 Conversion from Legacy Kirc} *)

(** Convert a legacy kirc_kernel to kernel_v2. Note: This creates a lazy IR that
    converts from Kirc_Ast when forced. *)
let of_kirc_kernel (kk : ('a, 'b, 'c) Kirc.kirc_kernel) ~name ~param_types :
    'a kernel_v2 =
  let ir = lazy (Sarek_ir.of_k_ext kk.Kirc.body) in
  let native_fn =
    match kk.Kirc.cpu_kern with
    | None -> None
    | Some fn ->
        Some
          (fun ~(block : Framework_sig.dims)
               ~(grid : Framework_sig.dims)
               args
             ->
            fn
              ~mode:Sarek_cpu_runtime.Parallel
              ~block:(block.x, block.y, block.z)
              ~grid:(grid.x, grid.y, grid.z)
              args)
  in
  {name; ir; native_fn; param_types; extensions = kk.Kirc.extensions}

(** Convert a legacy sarek_kernel to kernel_v2 *)
let of_sarek_kernel ((_spoc_k, kirc_k) : ('a, 'b, 'c, 'd, 'e) Kirc.sarek_kernel)
    ~name ~param_types : 'd kernel_v2 =
  of_kirc_kernel kirc_k ~name ~param_types

(** {1 Conversion to Legacy Kirc} *)

(** Convert kernel_v2 to legacy kirc_kernel. Note: This may force IR evaluation.
*)
let to_kirc_kernel (k : 'a kernel_v2) :
    (unit -> unit, unit, unit) Kirc.kirc_kernel =
  let body = Sarek_ir.to_k_ext (Lazy.force k.ir) in
  let cpu_kern =
    match k.native_fn with
    | None -> None
    | Some fn ->
        Some
          (fun ~mode:_ ~block ~grid args ->
            let bx, by, bz = block in
            let gx, gy, gz = grid in
            fn
              ~block:(Framework_sig.dims_3d bx by bz)
              ~grid:(Framework_sig.dims_3d gx gy gz)
              args)
  in
  {
    Kirc.ml_kern = (fun () -> ());
    Kirc.body;
    Kirc.body_v2 = None;
    (* ret_val is unused in V2 path; dummy value for type compatibility *)
    Kirc.ret_val = (Kirc_Ast.Empty, Obj.magic Spoc.Vector.int32);
    Kirc.extensions = k.extensions;
    Kirc.cpu_kern;
  }

(** {1 Execution} *)

(** Execute a kernel_v2 on a device with Obj.t array args. Note: Only works for
    Native backend. For JIT backends (CUDA/OpenCL), use run_with_args which
    provides properly typed arguments. *)
let run ~(device : Device.t) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem = 0) (k : 'a kernel_v2)
    (args : Obj.t array) : unit =
  ignore shared_mem ;
  match device.framework with
  | "Native" -> (
      match k.native_fn with
      | Some fn -> fn ~block ~grid args
      | None -> failwith ("Kernel " ^ k.name ^ " has no native function"))
  | fw ->
      failwith
        (Printf.sprintf
           "run with Obj.t array only works for Native backend; got %s. Use \
            run_with_args for JIT backends."
           fw)

(** Execute a kernel_v2 with explicit typed arguments. Works for all backends
    (Native, CUDA, OpenCL). *)
let run_with_args ~(device : Device.t) ~(block : Framework_sig.dims)
    ~(grid : Framework_sig.dims) ?(shared_mem = 0) (k : 'a kernel_v2)
    (args : Execute.arg list) : unit =
  match device.framework with
  | "Native" -> (
      match k.native_fn with
      | Some fn -> fn ~block ~grid (Execute.args_to_obj_array args)
      | None -> failwith "Native kernel has no native function")
  | "CUDA" ->
      let ir = Lazy.force k.ir in
      let source = Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir in
      Execute.run_typed
        ~device
        ~name:k.name
        ~source
        ~block
        ~grid
        ~shared_mem
        args
  | "OpenCL" ->
      let ir = Lazy.force k.ir in
      let source = Sarek_ir_opencl.generate_with_types ~types:ir.kern_types ir in
      Execute.run_typed
        ~device
        ~name:k.name
        ~source
        ~block
        ~grid
        ~shared_mem
        args
  | fw -> failwith ("Unsupported framework: " ^ fw)

(** {1 Utilities} *)

(** Get generated source for a specific backend *)
let source_for_backend (k : 'a kernel_v2) ~backend : string =
  let ir = Lazy.force k.ir in
  match backend with
  | "CUDA" -> Sarek_ir_cuda.generate_with_types ~types:ir.kern_types ir
  | "OpenCL" -> Sarek_ir_opencl.generate_with_types ~types:ir.kern_types ir
  | "Native" -> failwith "Native backend uses pre-compiled code, no source"
  | _ -> failwith ("Unknown backend: " ^ backend)

(** Check if kernel requires FP64 extension *)
let requires_fp64 k = Array.mem Kirc.ExFloat64 k.extensions

(** Check if kernel requires FP32 extension *)
let requires_fp32 k = Array.mem Kirc.ExFloat32 k.extensions

(** {1 Debugging} *)

(** Pretty-print the IR *)
let pp_ir fmt k =
  let ir = Lazy.force k.ir in
  Sarek_ir.pp_kernel fmt ir

(** Get IR as string *)
let ir_to_string k =
  let buf = Buffer.create 1024 in
  let fmt = Format.formatter_of_buffer buf in
  pp_ir fmt k ;
  Format.pp_print_flush fmt () ;
  Buffer.contents buf
