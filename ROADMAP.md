# SPOC/Sarek Roadmap

## Overview

This roadmap covers three major phases:
1. **Polymorphism** - Add polymorphic functions to Sarek kernels ✅ **COMPLETE**
2. **Recursion** - Support recursive functions via transformation ✅ **COMPLETE**
3. **SPOC/Sarek Rework** - Full rewrite with ctypes, plugin architecture

---

## Phase 1: Polymorphism ✅ COMPLETE

### Goal
Enable polymorphic functions in Sarek kernels that get monomorphized at compile time for GPU backends.

### Implementation Status
All steps completed:
- **Step 1.1** ✅ Type schemes in `Sarek_scheme.ml`
- **Step 1.2** ✅ Environment updated with `enter_level`/`exit_level`
- **Step 1.3** ✅ Generalization at let bindings in `Sarek_typer.ml`
- **Step 1.4** ✅ Monomorphization pass in `Sarek_mono.ml`
- **Step 1.5** ✅ Code generators updated for GPU and native

### Tests
- `test_polymorphism.exe` - inline polymorphic functions
- `test_module_poly.exe` - `[@sarek.module]` polymorphic functions

### Original State (before implementation)
- Type system uses `TVar` for unification (Sarek_types.ml:37-39)
- Level-based let-polymorphism infrastructure exists (enter_level/exit_level)
- No generalization step - all type variables resolve to concrete types

### Implementation Steps

#### Step 1.1: Add Type Schemes
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_types.ml`

```ocaml
(* Add after typ definition *)
type scheme = {
  quantified: int list;  (* IDs of generalized type variables *)
  body: typ;
}

(* Add generalization function *)
let generalize (level: int) (t: typ) : scheme =
  let free_tvars = ref [] in
  let rec collect t =
    match repr t with
    | TVar ({contents = Unbound (id, l)} as r) when l > level ->
        if not (List.mem id !free_tvars) then
          free_tvars := id :: !free_tvars;
        r := Unbound (id, level)  (* Lower the level *)
    | TVar {contents = Link t} -> collect t
    | TVec t | TArr (t, _) -> collect t
    | TFun (args, ret) -> List.iter collect args; collect ret
    | TRecord (_, fields) -> List.iter (fun (_, t) -> collect t) fields
    | TTuple ts -> List.iter collect ts
    | _ -> ()
  in
  collect t;
  { quantified = !free_tvars; body = t }

(* Instantiation: replace quantified vars with fresh ones *)
let instantiate (s: scheme) : typ =
  if s.quantified = [] then s.body
  else
    let subst = List.map (fun id -> (id, fresh_tvar ())) s.quantified in
    let rec inst t =
      match repr t with
      | TVar {contents = Unbound (id, _)} ->
          (try List.assoc id subst with Not_found -> t)
      | TVar {contents = Link t} -> inst t
      | TVec t -> TVec (inst t)
      | TArr (t, m) -> TArr (inst t, m)
      | TFun (args, ret) -> TFun (List.map inst args, inst ret)
      | TRecord (n, fields) -> TRecord (n, List.map (fun (f, t) -> (f, inst t)) fields)
      | TTuple ts -> TTuple (List.map inst ts)
      | _ -> t
    in
    inst s.body
```

#### Step 1.2: Update Environment for Schemes
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_env.ml`

```ocaml
(* Extend var_info to optionally hold a scheme *)
type var_info = {
  vi_type: typ;
  vi_scheme: scheme option;  (* NEW: for let-polymorphism *)
  vi_mutable: bool;
  vi_is_param: bool;
  vi_index: int;
  vi_is_vec: bool;
}

(* When looking up, instantiate if there's a scheme *)
let lookup_and_instantiate name env =
  match find_var name env with
  | Some vi ->
      let ty = match vi.vi_scheme with
        | Some s -> instantiate s
        | None -> vi.vi_type
      in
      Some { vi with vi_type = ty }
  | None -> None
```

#### Step 1.3: Generalize at Let Bindings
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_typer.ml`

```ocaml
(* In ELet case, after typing value, generalize if not mutable *)
| ELet (name, ty_annot, value, body) ->
    let env' = enter_level env in
    let* tv, env' = infer env' value in
    let* () = (* ... type annotation check ... *) in
    let var_id = fresh_var_id () in
    let level = get_level env in  (* Original level, not env' *)
    let scheme = generalize level tv.ty in
    let vi = {
      vi_type = tv.ty;
      vi_scheme = Some scheme;
      vi_mutable = false;
      (* ... *)
    } in
    let env'' = add_var name vi (exit_level env') in
    let* tb, env'' = infer env'' body in
    Ok (mk_texpr (TELet (name, var_id, tv, tb)) tb.ty loc, env'')
```

#### Step 1.4: Monomorphization Pass
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_mono.ml` (NEW)

Create a monomorphization pass that runs after type inference:

```ocaml
(* Collect all instantiation sites for polymorphic functions *)
type mono_instance = {
  name: string;
  orig_params: tparam list;
  specialized_types: typ list;
  body: texpr;
}

(* Walk the typed AST, collecting concrete instantiations *)
let collect_instances (kernel: tkernel) : (string * typ list) list = ...

(* Generate specialized function copies *)
let monomorphize (kernel: tkernel) : tkernel =
  let instances = collect_instances kernel in
  (* For each polymorphic function with multiple instances,
     create specialized copies with mangled names *)
  ...
```

#### Step 1.5: Update Code Generators
**Files:**
- `SpocLibs/Sarek_ppx/lib/Sarek_lower.ml`
- `SpocLibs/Sarek/Gen.ml`
- `SpocLibs/Sarek_ppx/lib/Sarek_native_gen.ml`

After monomorphization, all types are concrete. Code generators need minimal changes - they receive fully-typed, monomorphic code.

### Testing
```ocaml
(* Test: polymorphic identity *)
let%kernel identity_test = fun (a: float32 vector) (b: int32 vector) ->
  let id x = x in
  a.(0) <- id a.(0);
  b.(0) <- id b.(0)  (* Different instantiation *)

(* Test: polymorphic swap *)
let%kernel swap_test = fun (a: float32 vector) (i: int32) (j: int32) ->
  let swap arr x y =
    let tmp = arr.(x) in
    arr.(x) <- arr.(y);
    arr.(y) <- tmp
  in
  swap a i j
```

### Complexity Assessment
- **Total effort:** Medium
- **Risk:** Low - type system already has infrastructure
- **Dependencies:** None

---

## Phase 2: Recursion ✅ COMPLETE

### Goal
Support recursive functions by transforming tail recursion to loops and bounded recursion to inlining.

### Implementation Status
All steps completed:
- **Step 2.1** ✅ Parse recursive bindings - `MFun` has `is_rec` field
- **Step 2.2** ✅ Typed AST - `recursion_info` in `Sarek_tailrec.ml`
- **Step 2.3** ✅ Tail recursion detection - `is_tail_recursive` function
- **Step 2.4** ✅ Tail recursion elimination - transforms to while loops with temp vars
- **Step 2.5** ⏸️ Bounded recursion inlining - infrastructure exists but disabled (needs proper termination analysis)
- **Step 2.6** ✅ Validation during type checking
- **Step 2.7** ✅ Code generators work (mandelbrot benchmark passes)

### Tests
- `test_bounded_recursion.exe` - factorial, power, GCD (all tail-recursive)
- `test_mandelbrot.exe` - tail-recursive mandelbrot iteration

### Key Implementation Details
- Uses temporary variables to avoid sequential assignment issues (`gcd b (a mod b)`)
- Generates `_continue` flag and `_result` variable for loop control
- Loop variables prefixed with `__` to avoid C redeclaration issues

### Original State (before implementation)
- No recursion support
- Functions defined in `MFun` are non-recursive
- Code generators don't handle recursion

### Implementation Steps

#### Step 2.1: Parse Recursive Bindings
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_parse.ml`, `SpocLibs/Sarek_ppx/lib/Sarek_ast.ml`

```ocaml
(* In Sarek_ast.ml - add recursive marker *)
type module_item =
  | MConst of string * type_expr * expr
  | MFun of string * param list * expr
  | MRecFun of string * param list * expr  (* NEW: recursive function *)

(* In Sarek_parse.ml - detect `let rec f ...` *)
let parse_module_item item =
  match item with
  | { pstr_desc = Pstr_value (Recursive, [vb]); _ } ->
      MRecFun (name, params, body)
  | ...
```

#### Step 2.2: Typed AST for Recursion
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_typed_ast.ml`

```ocaml
type tmodule_item =
  | TMConst of string * int * typ * texpr
  | TMFun of string * tparam list * texpr
  | TMRecFun of string * tparam list * texpr * recursion_info

and recursion_info = {
  ri_is_tail: bool;           (* Entire function is tail-recursive *)
  ri_max_depth: int option;   (* Bounded depth if known *)
  ri_inline_limit: int;       (* Max iterations to inline *)
}
```

#### Step 2.3: Tail Recursion Detection
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_tailrec.ml` (NEW)

```ocaml
(* Analyze a function body for tail recursion *)
let is_tail_recursive (fname: string) (body: texpr) : bool =
  let rec check in_tail expr =
    match expr.te with
    | TEApp (fn, _) when is_self_call fname fn -> in_tail
    | TEIf (_, then_, Some else_) ->
        check true then_ && check true else_
    | TEIf (_, then_, None) ->
        check true then_
    | TESeq exprs ->
        let rec check_seq = function
          | [] -> true
          | [last] -> check true last
          | _ :: rest -> check_seq rest
        in
        check_seq exprs
    | TELet (_, _, _, body) | TELetMut (_, _, _, body) ->
        check true body
    | TEFor _ | TEWhile _ -> false  (* Recursive calls in loops not tail *)
    | _ -> true  (* Non-recursive expressions are fine *)
  in
  check true body
```

#### Step 2.4: Tail Recursion Elimination
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_tailrec.ml`

Transform tail-recursive functions to loops:

```ocaml
(* Transform:
   let rec f x y = if cond then f (x+1) (y-1) else x
   Into:
   let f x y =
     let x_loop = mut x in
     let y_loop = mut y in
     while true do
       if cond then begin
         x_loop := !x_loop + 1;
         y_loop := !y_loop - 1
       end else return !x_loop
     done
*)
let eliminate_tail_recursion (fname: string) (params: tparam list)
    (body: texpr) : texpr =
  (* 1. Create mutable loop variables for each parameter *)
  let loop_vars = List.map (fun p -> (p.tparam_name, fresh_var_id ())) params in

  (* 2. Replace recursive calls with assignments + continue *)
  let rec transform expr =
    match expr.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Replace with: p1 := arg1; p2 := arg2; ... (implicit continue) *)
        mk_seq (List.map2 (fun (name, id) arg ->
          mk_texpr (TEAssign (name, id, arg)) t_unit expr.te_loc
        ) loop_vars args)
    | TEIf (cond, then_, else_opt) ->
        { expr with te = TEIf (cond, transform then_,
                               Option.map transform else_opt) }
    | TESeq exprs ->
        { expr with te = TESeq (List.map transform exprs) }
    (* ... other cases ... *)
  in

  (* 3. Wrap in while-true loop *)
  mk_while_true (transform body)
```

#### Step 2.5: Bounded Recursion Inlining
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_inline.ml` (NEW)

For non-tail-recursive functions with bounded depth:

```ocaml
(* Inline a recursive function up to max_depth calls *)
let inline_bounded (fname: string) (params: tparam list)
    (body: texpr) (max_depth: int) : texpr =

  let rec inline depth expr =
    if depth > max_depth then
      mk_texpr (TEApp (mk_error_fn, [])) expr.ty expr.te_loc
    else match expr.te with
    | TEApp (fn, args) when is_self_call fname fn ->
        (* Inline the body with substituted arguments *)
        let subst = List.map2 (fun p a -> (p.tparam_name, a)) params args in
        substitute_and_inline (depth + 1) subst body
    | _ ->
        (* Recurse into subexpressions *)
        map_texpr (inline depth) expr
  in
  inline 0 body
```

#### Step 2.6: Validate Recursion Constraints
**Files:** `SpocLibs/Sarek_ppx/lib/Sarek_typer.ml`

Add validation during type checking:

```ocaml
(* In infer_kernel, after typing TMRecFun *)
| MRecFun (name, params, body) ->
    (* Type the function (add to env first for recursion) *)
    let fn_ty = TFun (param_types, fresh_tvar ()) in
    let env' = add_local_fun name fn_ty env in
    let* tbody, _ = infer env' body in

    (* Analyze recursion pattern *)
    let is_tail = Sarek_tailrec.is_tail_recursive name tbody in
    let max_depth = Sarek_tailrec.detect_bounded_depth name tbody in

    if not is_tail && max_depth = None then
      Error [Unsupported_recursion (name, loc)]
    else
      let ri = { ri_is_tail = is_tail; ri_max_depth = max_depth;
                 ri_inline_limit = 16 } in
      Ok (TMRecFun (name, tparams, tbody, ri) :: acc)
```

#### Step 2.7: Update Code Generators
After transformation, recursive functions become loops or inlined code. Backends need no special handling.

### Testing
```ocaml
(* Test: tail-recursive factorial *)
let%kernel factorial_test = fun (result: int32 vector) (n: int32) ->
  let rec fact_aux acc x =
    if x <= 1l then acc
    else fact_aux (acc * x) (x - 1l)
  in
  result.(0) <- fact_aux 1l n

(* Test: tail-recursive sum *)
let%kernel sum_test = fun (a: int32 vector) (n: int32) ->
  let rec sum_aux acc i =
    if i >= n then acc
    else sum_aux (acc + a.(i)) (i + 1l)
  in
  result.(0) <- sum_aux 0l 0l

(* Test: bounded recursion (binary tree depth <= 4) *)
let%kernel tree_test = fun (result: int32 vector) ->
  let rec depth_first node d =
    if d >= 4l then ()
    else begin
      visit node;
      depth_first (left node) (d + 1l);
      depth_first (right node) (d + 1l)
    end
  in
  depth_first root 0l
```

### Complexity Assessment
- **Total effort:** Medium-High
- **Risk:** Medium - tail recursion is straightforward, general recursion requires careful analysis
- **Dependencies:** None

---

## Phase 3: SPOC/Sarek Rework with Ctypes

### Goal
Complete rewrite of SPOC with:
1. Pure OCaml ctypes bindings (no C stubs)
2. Modular plugin architecture for GPU backends
3. Clean separation between Sarek (compiler) and SPOC (runtime)
4. Multiple opam packages

### Current State
- C stubs in `Spoc/*.c` (9 files, ~5000 lines of C)
- Mixed OCaml/C for CUDA and OpenCL
- Tight coupling between Sarek and SPOC

### Target Repository Structure
```
sarek/
├── sarek.opam                    # DSL compiler package
├── sarek-runtime.opam            # Core runtime (always required)
├── sarek-cuda.opam               # CUDA plugin (optional)
├── sarek-opencl.opam             # OpenCL plugin (optional)
├── dune-project
│
├── lib/                          # SAREK COMPILER
│   ├── ast/                      # AST definitions
│   │   ├── Sarek_ast.ml
│   │   ├── Sarek_typed_ast.ml
│   │   └── Sarek_ir.ml           # Kirc_Ast successor
│   ├── typing/                   # Type system
│   │   ├── Sarek_types.ml
│   │   ├── Sarek_env.ml
│   │   └── Sarek_typer.ml
│   ├── transform/                # IR transformations
│   │   ├── Sarek_lower.ml
│   │   ├── Sarek_tailrec.ml      # Recursion elimination
│   │   ├── Sarek_mono.ml         # Monomorphization
│   │   └── Sarek_inline.ml       # Function inlining
│   ├── backend/                  # Code generators
│   │   ├── Backend_sig.ml        # Backend interface
│   │   ├── Backend_cuda.ml
│   │   ├── Backend_opencl.ml
│   │   └── Backend_native.ml
│   ├── intrinsics/               # Intrinsic definitions
│   │   └── Sarek_intrinsics.ml
│   └── kernel/                   # Kernel type
│       └── Sarek_kernel.ml
│
├── ppx/                          # PPX preprocessor
│   ├── Sarek_ppx.ml
│   ├── Sarek_parse.ml
│   └── Sarek_quote.ml
│
├── runtime/                      # SPOC RUNTIME
│   ├── core/                     # Core abstractions
│   │   ├── Spoc.ml               # Main entry
│   │   ├── Device.ml             # Device abstraction
│   │   ├── Vector.ml             # GPU vectors
│   │   ├── Memory.ml             # Memory management
│   │   └── Kernel.ml             # Kernel execution
│   ├── framework/                # Plugin system
│   │   ├── Framework_sig.ml      # Plugin interface
│   │   └── Framework_registry.ml # Plugin registration
│   └── native/                   # CPU runtime
│       ├── Native_runtime.ml
│       └── Native_bsp.ml
│
└── plugins/                      # GPU PLUGINS
    ├── cuda/
    │   ├── Cuda_types.ml         # Ctypes type definitions
    │   ├── Cuda_bindings.ml      # Driver API bindings
    │   ├── Cuda_nvrtc.ml         # Runtime compilation
    │   ├── Cuda_api.ml           # High-level wrapper
    │   └── Cuda_plugin.ml        # Plugin implementation
    └── opencl/
        ├── Opencl_types.ml
        ├── Opencl_bindings.ml
        ├── Opencl_api.ml
        └── Opencl_plugin.ml
```

### Implementation Steps

#### Step 3.1: Create Framework Plugin Interface
**Files:** `runtime/framework/Framework_sig.ml`

```ocaml
module type Device_sig = sig
  type t
  type id

  val init : unit -> unit
  val count : unit -> int
  val get : int -> t
  val id : t -> id
  val name : t -> string

  type capabilities = {
    max_threads_per_block : int;
    max_block_dims : int * int * int;
    max_grid_dims : int * int * int;
    shared_mem_per_block : int;
    total_global_mem : int64;
    compute_capability : int * int;
    supports_fp64 : bool;
    supports_atomics : bool;
  }

  val capabilities : t -> capabilities
end

module type Memory_sig = sig
  type device
  type 'a buffer

  val alloc : device -> int -> 'a Bigarray.kind -> 'a buffer
  val free : 'a buffer -> unit
  val host_to_device : src:'a array1 -> dst:'a buffer -> unit
  val device_to_host : src:'a buffer -> dst:'a array1 -> unit
end

module type Kernel_sig = sig
  type device
  type t
  type args

  val compile : device -> name:string -> source:string -> t
  val create_args : unit -> args
  val set_arg_buffer : args -> int -> _ buffer -> unit
  val set_arg_int32 : args -> int -> int32 -> unit
  val set_arg_float : args -> int -> float -> unit
  val launch : t -> args:args -> grid:dims -> block:dims ->
               shared_mem:int -> stream:stream option -> unit
end

module type S = sig
  val name : string
  val version : int * int * int

  module Device : Device_sig
  module Memory : Memory_sig with type device := Device.t
  module Kernel : Kernel_sig with type device := Device.t
  module Stream : Stream_sig with type device := Device.t
  module Event : Event_sig
end
```

#### Step 3.2: Create Plugin Registry
**Files:** `runtime/framework/Framework_registry.ml`

```ocaml
type registered = {
  name : string;
  plugin : (module Framework_sig.S);
  priority : int;
}

let plugins : (string, registered) Hashtbl.t = Hashtbl.create 8

let register ?(priority = 0) (module P : Framework_sig.S) =
  Hashtbl.replace plugins P.name {
    name = P.name;
    plugin = (module P);
    priority
  }

let find name =
  Option.map (fun r -> r.plugin) (Hashtbl.find_opt plugins name)

let all () =
  Hashtbl.to_seq_values plugins
  |> List.of_seq
  |> List.sort (fun a b -> compare b.priority a.priority)
  |> List.map (fun r -> r.plugin)
```

#### Step 3.3: CUDA Ctypes Bindings
**Files:** `plugins/cuda/Cuda_types.ml`

```ocaml
open Ctypes

(* Device pointer - 64-bit address *)
type cu_deviceptr = Unsigned.uint64
let cu_deviceptr = uint64_t

(* Opaque handles *)
type cu_context
let cu_context : cu_context structure typ = structure "CUctx_st"
let cu_context_ptr = ptr cu_context

type cu_module
let cu_module : cu_module structure typ = structure "CUmod_st"
let cu_module_ptr = ptr cu_module

type cu_function
let cu_function : cu_function structure typ = structure "CUfunc_st"
let cu_function_ptr = ptr cu_function

type cu_stream
let cu_stream : cu_stream structure typ = structure "CUstream_st"
let cu_stream_ptr = ptr cu_stream

(* Result codes *)
type cu_result =
  | CUDA_SUCCESS
  | CUDA_ERROR_INVALID_VALUE
  | CUDA_ERROR_OUT_OF_MEMORY
  | CUDA_ERROR_NOT_INITIALIZED
  (* ... more error codes ... *)
  | CUDA_ERROR_UNKNOWN of int

let cu_result = view
  ~read:(function 0 -> CUDA_SUCCESS | 1 -> CUDA_ERROR_INVALID_VALUE | ...)
  ~write:(function CUDA_SUCCESS -> 0 | ...)
  int
```

**Files:** `plugins/cuda/Cuda_bindings.ml`

```ocaml
open Ctypes
open Foreign

(* Dynamic library loading *)
let cuda_lib =
  try Dl.dlopen ~filename:"libcuda.so" ~flags:[Dl.RTLD_LAZY; Dl.RTLD_GLOBAL]
  with _ ->
    try Dl.dlopen ~filename:"libcuda.dylib" ~flags:[Dl.RTLD_LAZY]
    with _ -> failwith "CUDA library not found"

let foreign_cuda name typ = foreign ~from:cuda_lib name typ

(* Driver API *)
let cuInit =
  foreign_cuda "cuInit" (int @-> returning cu_result)

let cuDeviceGetCount =
  foreign_cuda "cuDeviceGetCount" (ptr int @-> returning cu_result)

let cuDeviceGet =
  foreign_cuda "cuDeviceGet" (ptr int @-> int @-> returning cu_result)

let cuDeviceGetName =
  foreign_cuda "cuDeviceGetName" (ptr char @-> int @-> int @-> returning cu_result)

let cuCtxCreate =
  foreign_cuda "cuCtxCreate_v2"
    (ptr cu_context_ptr @-> int @-> int @-> returning cu_result)

let cuMemAlloc =
  foreign_cuda "cuMemAlloc_v2"
    (ptr cu_deviceptr @-> size_t @-> returning cu_result)

let cuMemFree =
  foreign_cuda "cuMemFree_v2" (cu_deviceptr @-> returning cu_result)

let cuMemcpyHtoD =
  foreign_cuda "cuMemcpyHtoD_v2"
    (cu_deviceptr @-> ptr void @-> size_t @-> returning cu_result)

let cuMemcpyDtoH =
  foreign_cuda "cuMemcpyDtoH_v2"
    (ptr void @-> cu_deviceptr @-> size_t @-> returning cu_result)

let cuModuleLoadData =
  foreign_cuda "cuModuleLoadData"
    (ptr cu_module_ptr @-> ptr void @-> returning cu_result)

let cuModuleGetFunction =
  foreign_cuda "cuModuleGetFunction"
    (ptr cu_function_ptr @-> cu_module_ptr @-> string @-> returning cu_result)

let cuLaunchKernel =
  foreign_cuda "cuLaunchKernel" (
    cu_function_ptr @->
    uint @-> uint @-> uint @->  (* grid dims *)
    uint @-> uint @-> uint @->  (* block dims *)
    uint @->                     (* shared mem *)
    cu_stream_ptr @->           (* stream *)
    ptr (ptr void) @->          (* kernel params *)
    ptr (ptr void) @->          (* extra *)
    returning cu_result
  )
```

**Files:** `plugins/cuda/Cuda_nvrtc.ml`

```ocaml
(* NVRTC runtime compilation bindings *)
let nvrtc_lib =
  Dl.dlopen ~filename:"libnvrtc.so" ~flags:[Dl.RTLD_LAZY]

let nvrtcCreateProgram =
  foreign ~from:nvrtc_lib "nvrtcCreateProgram" (
    ptr nvrtc_program_ptr @->
    string @->                  (* source *)
    string @->                  (* name *)
    int @->                     (* numHeaders *)
    ptr string @->              (* headers *)
    ptr string @->              (* includeNames *)
    returning nvrtc_result
  )

let nvrtcCompileProgram =
  foreign ~from:nvrtc_lib "nvrtcCompileProgram" (
    nvrtc_program_ptr @->
    int @->                     (* numOptions *)
    ptr string @->              (* options *)
    returning nvrtc_result
  )

let nvrtcGetPTX =
  foreign ~from:nvrtc_lib "nvrtcGetPTX" (
    nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result
  )
```

#### Step 3.4: CUDA Plugin Implementation
**Files:** `plugins/cuda/Cuda_plugin.ml`

```ocaml
module Cuda : Framework_sig.S = struct
  let name = "CUDA"
  let version = (12, 0, 0)

  module Device = struct
    type t = {
      id : int;
      context : cu_context_ptr;
      name : string;
      caps : capabilities;
    }

    let init () = check "cuInit" (Cuda_bindings.cuInit 0)

    let count () =
      let n = allocate int 0 in
      check "cuDeviceGetCount" (cuDeviceGetCount n);
      !@ n

    let get idx =
      let dev = allocate int 0 in
      check "cuDeviceGet" (cuDeviceGet dev idx);
      (* ... get name, capabilities ... *)
      { id = idx; context; name; caps }
  end

  module Memory = struct
    type 'a buffer = {
      ptr : cu_deviceptr;
      size : int;
      device : Device.t;
    }

    let alloc device size kind =
      let bytes = size * Bigarray.kind_size_in_bytes kind in
      let ptr = allocate cu_deviceptr Unsigned.UInt64.zero in
      check "cuMemAlloc" (cuMemAlloc ptr (Unsigned.Size_t.of_int bytes));
      { ptr = !@ ptr; size; device }

    let host_to_device ~src ~dst =
      let src_ptr = bigarray_start array1 src |> to_voidp in
      let bytes = Bigarray.Array1.size_in_bytes src in
      check "cuMemcpyHtoD" (cuMemcpyHtoD dst.ptr src_ptr
        (Unsigned.Size_t.of_int bytes))
    (* ... *)
  end

  module Kernel = struct
    type t = {
      module_ : cu_module_ptr;
      function_ : cu_function_ptr;
    }

    let compile device ~name ~source =
      (* Use NVRTC to compile *)
      let ptx = Cuda_nvrtc.compile source name device.caps in
      (* Load module *)
      let module_ = allocate cu_module_ptr (from_voidp cu_module null) in
      check "cuModuleLoadData" (cuModuleLoadData module_ ptx);
      (* Get function *)
      let func = allocate cu_function_ptr (from_voidp cu_function null) in
      check "cuModuleGetFunction" (cuModuleGetFunction func (!@ module_) name);
      { module_ = !@ module_; function_ = !@ func }

    let launch kernel ~args ~grid ~block ~shared_mem ~stream =
      check "cuLaunchKernel" (cuLaunchKernel
        kernel.function_
        grid.x grid.y grid.z
        block.x block.y block.z
        shared_mem stream args null)
  end

  (* Stream, Event modules... *)
end

(* Auto-register *)
let () = Framework_registry.register ~priority:100 (module Cuda)
```

#### Step 3.5: OpenCL Ctypes Bindings
**Files:** `plugins/opencl/Opencl_bindings.ml`

```ocaml
let opencl_lib =
  Dl.dlopen ~filename:"libOpenCL.so" ~flags:[Dl.RTLD_LAZY]

let clGetPlatformIDs =
  foreign ~from:opencl_lib "clGetPlatformIDs"
    (uint32_t @-> ptr cl_platform_id @-> ptr uint32_t @-> returning cl_int)

let clGetDeviceIDs =
  foreign ~from:opencl_lib "clGetDeviceIDs"
    (cl_platform_id @-> uint64_t @-> uint32_t @->
     ptr cl_device_id @-> ptr uint32_t @-> returning cl_int)

let clCreateContext =
  foreign ~from:opencl_lib "clCreateContext"
    (ptr void @-> uint32_t @-> ptr cl_device_id @->
     ptr void @-> ptr void @-> ptr cl_int @-> returning cl_context)

let clCreateBuffer =
  foreign ~from:opencl_lib "clCreateBuffer"
    (cl_context @-> uint64_t @-> size_t @-> ptr void @->
     ptr cl_int @-> returning cl_mem)

let clCreateProgramWithSource =
  foreign ~from:opencl_lib "clCreateProgramWithSource"
    (cl_context @-> uint32_t @-> ptr string @-> ptr size_t @->
     ptr cl_int @-> returning cl_program)

let clBuildProgram =
  foreign ~from:opencl_lib "clBuildProgram"
    (cl_program @-> uint32_t @-> ptr cl_device_id @-> string @->
     ptr void @-> ptr void @-> returning cl_int)

let clEnqueueNDRangeKernel =
  foreign ~from:opencl_lib "clEnqueueNDRangeKernel"
    (cl_command_queue @-> cl_kernel @-> uint32_t @->
     ptr size_t @-> ptr size_t @-> ptr size_t @->
     uint32_t @-> ptr cl_event @-> ptr cl_event @-> returning cl_int)
```

#### Step 3.6: Unified Device Abstraction
**Files:** `runtime/core/Device.ml`

```ocaml
type t = {
  id : int;
  name : string;
  framework : string;
  plugin : (module Framework_sig.S);
  handle : Obj.t;  (* Plugin-specific handle *)
}

let init ?(frameworks = ["CUDA"; "OpenCL"; "Native"]) () : t array =
  let devices = ref [] in
  let device_id = ref 0 in

  frameworks |> List.iter (fun fw ->
    match Framework_registry.find fw with
    | None -> ()
    | Some (module P) ->
        P.Device.init ();
        for i = 0 to P.Device.count () - 1 do
          let d = P.Device.get i in
          devices := {
            id = !device_id;
            name = P.Device.name d;
            framework = P.name;
            plugin = (module P);
            handle = Obj.repr d;
          } :: !devices;
          incr device_id
        done
  );

  Array.of_list (List.rev !devices)
```

#### Step 3.7: Kernel Execution
**Files:** `runtime/core/Kernel.ml`

```ocaml
let run (kernel : _ Sarek_kernel.t) ~device ~block ~grid args =
  match device.framework with
  | "Native" | "Interpreter" ->
      (match kernel.cpu_kern with
       | Some f -> f ~parallel:true ~block ~grid args
       | None -> failwith "No CPU implementation")
  | "CUDA" | "OpenCL" ->
      let module P = (val device.plugin) in
      (* Generate source if not cached *)
      let source = Backend.generate device.framework kernel.ir in
      (* Compile *)
      let compiled = P.Kernel.compile (Obj.obj device.handle)
        ~name:kernel.name ~source in
      (* Bind arguments *)
      let kargs = P.Kernel.create_args () in
      bind_args kargs kernel.param_types args;
      (* Launch *)
      P.Kernel.launch compiled ~args:kargs ~grid ~block
        ~shared_mem:0 ~stream:None
  | _ -> failwith "Unknown framework"
```

#### Step 3.8: Migration Strategy

**Phase 3.8.1: Parallel Development**
- Create new `plugins/` directory alongside existing `Spoc/`
- Build ctypes bindings without removing C stubs
- Validate bindings work with simple test

**Phase 3.8.2: Feature Parity**
- Implement all functionality from C stubs in ctypes
- Add comprehensive tests comparing results
- Benchmark performance (ctypes overhead should be negligible)

**Phase 3.8.3: Switch Over**
- Update all imports to use new plugin system
- Remove C stubs
- Update build system (no more C compilation)

**Phase 3.8.4: Package Split**
- Create separate opam packages
- Ensure optional dependencies work correctly
- Test installation without CUDA/OpenCL

### Dune Configuration

```lisp
; plugins/cuda/dune
(library
 (name sarek_cuda)
 (public_name sarek-cuda)
 (libraries sarek-runtime ctypes ctypes-foreign)
 ; No foreign_stubs needed!
)

; runtime/dune
(library
 (name sarek_runtime)
 (public_name sarek-runtime)
 (libraries bigarray unix)
)
```

### Testing Plan

```ocaml
(* Test CUDA ctypes bindings *)
let%test "cuda_device_init" =
  Cuda.Device.init ();
  Cuda.Device.count () >= 0

let%test "cuda_memory_roundtrip" =
  let device = Cuda.Device.get 0 in
  let buf = Cuda.Memory.alloc device 100 Bigarray.float32 in
  let host = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 100 in
  for i = 0 to 99 do host.{i} <- float i done;
  Cuda.Memory.host_to_device ~src:host ~dst:buf;
  let result = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 100 in
  Cuda.Memory.device_to_host ~src:buf ~dst:result;
  Cuda.Memory.free buf;
  host = result

(* Test plugin registry *)
let%test "framework_registration" =
  Framework_registry.find "CUDA" <> None

(* Test unified device API *)
let%test "unified_device_init" =
  let devices = Device.init () in
  Array.length devices > 0
```

### Complexity Assessment
- **Total effort:** High
- **Risk:** Medium - ctypes is well-tested, but full CUDA/OpenCL API is large
- **Dependencies:** ctypes, ctypes-foreign

### Benefits Summary

| Aspect | C Stubs | Ctypes |
|--------|---------|--------|
| Maintenance | Two languages | Pure OCaml |
| Type safety | Manual | Automatic |
| Build | Complex (C compiler) | Simple |
| Cross-platform | Separate stubs | Same code |
| Error messages | C errors | OCaml exceptions |
| Distribution | Needs C toolchain | Pure OCaml |

---

## Timeline Summary

| Phase | Effort | Status |
|-------|--------|--------|
| 1. Polymorphism | Medium | ✅ Complete |
| 2. Recursion | Medium-High | ✅ Complete |
| 3. SPOC Rework | High | Not started |

Next steps:
1. Phase 3: SPOC Rework with ctypes plugin architecture

---

## Open Questions

1. **Polymorphism scope**: Should we support higher-kinded types or just rank-1?
2. **Recursion limits**: What's a reasonable default inline limit for bounded recursion?
3. **Vulkan/Metal**: Should plugins for these be included in initial scope?
4. **WebGPU**: Browser support via wasm - worth considering?
