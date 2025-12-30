(******************************************************************************
 * Sarek Runtime - Kernel Execution
 *
 * Provides unified kernel compilation and execution across backends.
 ******************************************************************************)

open Sarek_framework

(** Compiled kernel handle *)
type t = {
  device : Device.t;
  name : string;
  handle : Obj.t;  (** Backend-specific kernel handle *)
}

(** Kernel arguments builder *)
type args = {
  device : Device.t;
  handle : Obj.t;  (** Backend-specific args handle *)
}

(** Compile a kernel from source *)
let compile (device : Device.t) ~(name : string) ~(source : string) : t =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let k = B.Kernel.compile dev ~name ~source in
      {device; name; handle = Obj.repr k}

(** Compile with caching *)
let compile_cached (device : Device.t) ~(name : string) ~(source : string) : t =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let dev = B.Device.get device.backend_id in
      let k = B.Kernel.compile_cached dev ~name ~source in
      {device; name; handle = Obj.repr k}

(** Create an arguments object *)
let create_args (device : Device.t) : args =
  match Framework_registry.find_backend device.framework with
  | None -> failwith ("Unknown framework: " ^ device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let a = B.Kernel.create_args () in
      {device; handle = Obj.repr a}

(** Set buffer argument *)
let set_arg_buffer (args : args) (idx : int) (buf : _ Memory.buffer) : unit =
  match Framework_registry.find_backend args.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let a : B.Kernel.args = Obj.obj args.handle in
      let b : _ B.Memory.buffer = Obj.obj buf.Memory.handle in
      B.Kernel.set_arg_buffer a idx b

(** Set int32 argument *)
let set_arg_int32 (args : args) (idx : int) (v : int32) : unit =
  match Framework_registry.find_backend args.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let a : B.Kernel.args = Obj.obj args.handle in
      B.Kernel.set_arg_int32 a idx v

(** Set int64 argument *)
let set_arg_int64 (args : args) (idx : int) (v : int64) : unit =
  match Framework_registry.find_backend args.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let a : B.Kernel.args = Obj.obj args.handle in
      B.Kernel.set_arg_int64 a idx v

(** Set float32 argument *)
let set_arg_float32 (args : args) (idx : int) (v : float) : unit =
  match Framework_registry.find_backend args.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let a : B.Kernel.args = Obj.obj args.handle in
      B.Kernel.set_arg_float32 a idx v

(** Set float64 argument *)
let set_arg_float64 (args : args) (idx : int) (v : float) : unit =
  match Framework_registry.find_backend args.device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) ->
      let a : B.Kernel.args = Obj.obj args.handle in
      B.Kernel.set_arg_float64 a idx v

(** Launch a kernel *)
let launch (kernel : t) ~(args : args) ~(grid : Framework_sig.dims)
    ~(block : Framework_sig.dims) ?(shared_mem = 0) () : unit =
  match Framework_registry.find_backend kernel.device.framework with
  | None -> failwith ("Unknown framework: " ^ kernel.device.framework)
  | Some (module B : Framework_sig.BACKEND) ->
      let k : B.Kernel.t = Obj.obj kernel.handle in
      let a : B.Kernel.args = Obj.obj args.handle in
      B.Kernel.launch k ~args:a ~grid ~block ~shared_mem ~stream:None

(** Clear all kernel caches *)
let clear_cache (device : Device.t) : unit =
  match Framework_registry.find_backend device.framework with
  | None -> ()
  | Some (module B : Framework_sig.BACKEND) -> B.Kernel.clear_cache ()
