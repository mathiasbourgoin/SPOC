(** Sarek_interp - CPU interpreter for Sarek kernels

    Executes Kirc_Ast on CPU for debugging and testing without GPU. Supports
    BSP-style barrier synchronization using OCaml 5 effects. *)

open Kirc_Ast

(** {1 BSP Barrier Effect}

    Used for synchronizing threads at barriers. Each thread is suspended when it
    hits a barrier, and all threads are resumed together. *)

type _ Effect.t += Barrier : unit Effect.t

(** {1 Runtime Values} *)

type value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array  (** type_name, fields *)
  | VConstr of string * int * value list  (** type, tag, args *)

(** {1 Thread State} *)

type thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** {1 Environment} *)

type env = {
  locals : (int, value) Hashtbl.t;  (** var_id -> value *)
  arrays : (string, value array) Hashtbl.t;  (** array_name -> data *)
  shared : (string, value array) Hashtbl.t;  (** shared arrays for block *)
  mutable functions : (string, k_ext) Hashtbl.t;  (** global functions *)
}

let create_env () =
  {
    locals = Hashtbl.create 32;
    arrays = Hashtbl.create 16;
    shared = Hashtbl.create 8;
    functions = Hashtbl.create 8;
  }

let copy_env env =
  {
    locals = Hashtbl.copy env.locals;
    arrays = env.arrays;
    (* shared across threads *)
    shared = env.shared;
    (* shared within block *)
    functions = env.functions;
  }

(** {1 Value Operations} *)

let to_int32 = function
  | VInt32 n -> n
  | VInt64 n -> Int64.to_int32 n
  | VFloat32 f -> Int32.of_float f
  | VFloat64 f -> Int32.of_float f
  | VBool b -> if b then 1l else 0l
  | _ -> failwith "to_int32: not a numeric value"

let to_int = function
  | VInt32 n -> Int32.to_int n
  | VInt64 n -> Int64.to_int n
  | VFloat32 f -> int_of_float f
  | VFloat64 f -> int_of_float f
  | VBool b -> if b then 1 else 0
  | _ -> failwith "to_int: not a numeric value"

let to_float32 = function
  | VFloat32 f -> f
  | VFloat64 f -> f
  | VInt32 n -> Int32.to_float n
  | VInt64 n -> Int64.to_float n
  | _ -> failwith "to_float32: not a numeric value"

let to_float64 = function
  | VFloat64 f -> f
  | VFloat32 f -> f
  | VInt32 n -> Int32.to_float n
  | VInt64 n -> Int64.to_float n
  | _ -> failwith "to_float64: not a numeric value"

let to_int64 = function
  | VInt64 n -> n
  | VInt32 n -> Int64.of_int32 n
  | VFloat32 f -> Int64.of_float f
  | VFloat64 f -> Int64.of_float f
  | VBool b -> if b then 1L else 0L
  | _ -> failwith "to_int64: not a numeric value"

let to_bool = function
  | VBool b -> b
  | VInt32 n -> n <> 0l
  | VInt64 n -> n <> 0L
  | VFloat32 f -> f <> 0.0
  | VFloat64 f -> f <> 0.0
  | _ -> failwith "to_bool: not convertible to bool"

let add_int32 a b = VInt32 (Int32.add (to_int32 a) (to_int32 b))

let sub_int32 a b = VInt32 (Int32.sub (to_int32 a) (to_int32 b))

let mul_int32 a b = VInt32 (Int32.mul (to_int32 a) (to_int32 b))

let div_int32 a b = VInt32 (Int32.div (to_int32 a) (to_int32 b))

let mod_int32 a b = VInt32 (Int32.rem (to_int32 a) (to_int32 b))

let add_float a b = VFloat32 (to_float32 a +. to_float32 b)

let sub_float a b = VFloat32 (to_float32 a -. to_float32 b)

let mul_float a b = VFloat32 (to_float32 a *. to_float32 b)

let div_float a b = VFloat32 (to_float32 a /. to_float32 b)

let eq_val a b =
  match (a, b) with
  | VInt32 x, VInt32 y -> x = y
  | VInt64 x, VInt64 y -> x = y
  | VFloat32 x, VFloat32 y -> x = y
  | VFloat64 x, VFloat64 y -> x = y
  | VBool x, VBool y -> x = y
  | _ -> to_int32 a = to_int32 b

let lt_val a b =
  match (a, b) with
  | VFloat32 x, VFloat32 y -> x < y
  | VFloat64 x, VFloat64 y -> x < y
  | VFloat32 x, VFloat64 y -> Float.of_int (int_of_float x) < y
  | VFloat64 x, VFloat32 y -> x < Float.of_int (int_of_float y)
  | VInt32 _, VFloat32 y -> to_float32 a < y
  | VFloat32 x, VInt32 _ -> x < to_float32 b
  | _ -> to_int32 a < to_int32 b

let gt_val a b =
  match (a, b) with
  | VFloat32 x, VFloat32 y -> x > y
  | VFloat64 x, VFloat64 y -> x > y
  | VFloat32 x, VFloat64 y -> Float.of_int (int_of_float x) > y
  | VFloat64 x, VFloat32 y -> x > Float.of_int (int_of_float y)
  | VInt32 _, VFloat32 y -> to_float32 a > y
  | VFloat32 x, VInt32 _ -> x > to_float32 b
  | _ -> to_int32 a > to_int32 b

let lte_val a b =
  match (a, b) with
  | VFloat32 x, VFloat32 y -> x <= y
  | VFloat64 x, VFloat64 y -> x <= y
  | VFloat32 x, VFloat64 y -> Float.of_int (int_of_float x) <= y
  | VFloat64 x, VFloat32 y -> x <= Float.of_int (int_of_float y)
  | VInt32 _, VFloat32 y -> to_float32 a <= y
  | VFloat32 x, VInt32 _ -> x <= to_float32 b
  | _ -> to_int32 a <= to_int32 b

let gte_val a b =
  match (a, b) with
  | VFloat32 x, VFloat32 y -> x >= y
  | VFloat64 x, VFloat64 y -> x >= y
  | VFloat32 x, VFloat64 y -> Float.of_int (int_of_float x) >= y
  | VFloat64 x, VFloat32 y -> x >= Float.of_int (int_of_float y)
  | VInt32 _, VFloat32 y -> to_float32 a >= y
  | VFloat32 x, VInt32 _ -> x >= to_float32 b
  | _ -> to_int32 a >= to_int32 b

(** {1 Intrinsics} *)

(** Check if path matches a GPU intrinsic path *)
let is_gpu_path = function
  | ["Gpu"] | [] | ["Sarek_stdlib"; "Gpu"] -> true
  | _ -> false

(** Check if path matches Float32 module *)
let is_float32_path = function
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] -> true
  | _ -> false

(** Check if path matches Float64 module *)
let is_float64_path = function
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] -> true
  | _ -> false

(** Check if path matches Int32 module *)
let is_int32_path = function
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> true
  | _ -> false

(** Check if path matches Std module *)
let is_std_path = function
  | ["Std"] | ["Gpu"] | [] | ["Sarek_stdlib"; "Std"] | ["Sarek_stdlib"; "Gpu"]
    ->
      true
  | _ -> false

let eval_intrinsic state path name args =
  match (path, name) with
  (* Thread indices *)
  | path, "thread_idx_x" when is_gpu_path path ->
      let x, _, _ = state.thread_idx in
      VInt32 (Int32.of_int x)
  | path, "thread_idx_y" when is_gpu_path path ->
      let _, y, _ = state.thread_idx in
      VInt32 (Int32.of_int y)
  | path, "thread_idx_z" when is_gpu_path path ->
      let _, _, z = state.thread_idx in
      VInt32 (Int32.of_int z)
  (* Block indices *)
  | path, "block_idx_x" when is_gpu_path path ->
      let x, _, _ = state.block_idx in
      VInt32 (Int32.of_int x)
  | path, "block_idx_y" when is_gpu_path path ->
      let _, y, _ = state.block_idx in
      VInt32 (Int32.of_int y)
  | path, "block_idx_z" when is_gpu_path path ->
      let _, _, z = state.block_idx in
      VInt32 (Int32.of_int z)
  (* Block dimensions *)
  | path, "block_dim_x" when is_gpu_path path ->
      let x, _, _ = state.block_dim in
      VInt32 (Int32.of_int x)
  | path, "block_dim_y" when is_gpu_path path ->
      let _, y, _ = state.block_dim in
      VInt32 (Int32.of_int y)
  | path, "block_dim_z" when is_gpu_path path ->
      let _, _, z = state.block_dim in
      VInt32 (Int32.of_int z)
  (* Grid dimensions *)
  | path, "grid_dim_x" when is_gpu_path path ->
      let x, _, _ = state.grid_dim in
      VInt32 (Int32.of_int x)
  | path, "grid_dim_y" when is_gpu_path path ->
      let _, y, _ = state.grid_dim in
      VInt32 (Int32.of_int y)
  | path, "grid_dim_z" when is_gpu_path path ->
      let _, _, z = state.grid_dim in
      VInt32 (Int32.of_int z)
  (* Global index helpers *)
  | path, "global_idx" when is_gpu_path path ->
      let tx, _, _ = state.thread_idx in
      let bx, _, _ = state.block_idx in
      let bdx, _, _ = state.block_dim in
      VInt32 (Int32.of_int ((bx * bdx) + tx))
  | path, "global_size" when is_gpu_path path ->
      let bdx, _, _ = state.block_dim in
      let gdx, _, _ = state.grid_dim in
      VInt32 (Int32.of_int (bdx * gdx))
  (* Barriers - perform effect for BSP synchronization *)
  | path, "block_barrier" when is_gpu_path path ->
      Effect.perform Barrier ;
      VUnit
  | path, "warp_barrier" when is_gpu_path path ->
      Effect.perform Barrier ;
      VUnit
  (* Math intrinsics - Float32 *)
  | path, "sin" when is_float32_path path ->
      VFloat32 (sin (to_float32 (List.hd args)))
  | path, "cos" when is_float32_path path ->
      VFloat32 (cos (to_float32 (List.hd args)))
  | path, "tan" when is_float32_path path ->
      VFloat32 (tan (to_float32 (List.hd args)))
  | path, "sqrt" when is_float32_path path ->
      VFloat32 (sqrt (to_float32 (List.hd args)))
  | path, "exp" when is_float32_path path ->
      VFloat32 (exp (to_float32 (List.hd args)))
  | path, "log" when is_float32_path path ->
      VFloat32 (log (to_float32 (List.hd args)))
  | path, "log10" when is_float32_path path ->
      VFloat32 (log10 (to_float32 (List.hd args)))
  | path, "pow" when is_float32_path path ->
      let base = to_float32 (List.nth args 0) in
      let exp = to_float32 (List.nth args 1) in
      VFloat32 (base ** exp)
  | path, "abs" when is_float32_path path ->
      VFloat32 (abs_float (to_float32 (List.hd args)))
  | path, "floor" when is_float32_path path ->
      VFloat32 (floor (to_float32 (List.hd args)))
  | path, "ceil" when is_float32_path path ->
      VFloat32 (ceil (to_float32 (List.hd args)))
  | path, "fma" when is_float32_path path ->
      let a = to_float32 (List.nth args 0) in
      let b = to_float32 (List.nth args 1) in
      let c = to_float32 (List.nth args 2) in
      VFloat32 ((a *. b) +. c)
  | path, "min" when is_float32_path path ->
      VFloat32
        (min (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | path, "max" when is_float32_path path ->
      VFloat32
        (max (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | path, "add" when is_float32_path path ->
      VFloat32 (to_float32 (List.nth args 0) +. to_float32 (List.nth args 1))
  | path, "sub" when is_float32_path path ->
      VFloat32 (to_float32 (List.nth args 0) -. to_float32 (List.nth args 1))
  | path, "mul" when is_float32_path path ->
      VFloat32 (to_float32 (List.nth args 0) *. to_float32 (List.nth args 1))
  | path, "div" when is_float32_path path ->
      VFloat32 (to_float32 (List.nth args 0) /. to_float32 (List.nth args 1))
  | path, "of_int" when is_float32_path path ->
      VFloat32 (Int32.to_float (to_int32 (List.hd args)))
  | path, "of_float64" when is_float32_path path ->
      VFloat32 (to_float64 (List.hd args))
  (* Float64 intrinsics *)
  | path, "of_int" when is_float64_path path ->
      VFloat64 (Int32.to_float (to_int32 (List.hd args)))
  | path, "of_float32" when is_float64_path path ->
      VFloat64 (to_float32 (List.hd args))
  (* Int32 math *)
  | path, "abs" when is_int32_path path ->
      VInt32 (Int32.abs (to_int32 (List.hd args)))
  | path, "min" when is_int32_path path ->
      VInt32 (min (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | path, "max" when is_int32_path path ->
      VInt32 (max (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  (* Type conversions *)
  | path, "float" when is_std_path path ->
      VFloat32 (Int32.to_float (to_int32 (List.hd args)))
  | path, "float64" when is_std_path path ->
      VFloat64 (Int32.to_float (to_int32 (List.hd args)))
  | path, "int_of_float" when is_std_path path ->
      VInt32 (Int32.of_float (to_float32 (List.hd args)))
  | path, "int_of_float64" when is_std_path path ->
      VInt32 (Int32.of_float (to_float64 (List.hd args)))
  (* Bitwise operations (via intrinsics with format strings) *)
  | _, "(%s ^ %s)" ->
      VInt32
        (Int32.logxor (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | _, "(%s & %s)" ->
      VInt32
        (Int32.logand (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | _, "(%s | %s)" ->
      VInt32
        (Int32.logor (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | _, "(%s << %s)" ->
      VInt32
        (Int32.shift_left
           (to_int32 (List.nth args 0))
           (to_int (List.nth args 1)))
  | _, "(%s >> %s)" ->
      VInt32
        (Int32.shift_right_logical
           (to_int32 (List.nth args 0))
           (to_int (List.nth args 1)))
  (* Atomic operations - interpreter runs sequentially so these are just regular ops *)
  | path, "atomic_add_int32" when is_gpu_path path -> (
      (* args: array, index, value - returns old value, adds value to array[index] *)
      match args with
      | [VArray arr; idx_val; add_val] ->
          let idx = Int32.to_int (to_int32 idx_val) in
          let old_val = arr.(idx) in
          arr.(idx) <- add_int32 old_val add_val ;
          old_val
      | _ -> failwith "atomic_add_int32: wrong arguments")
  | path, "atomic_add_global_int32" when is_gpu_path path -> (
      (* Same as above but for global memory *)
      match args with
      | [VArray arr; idx_val; add_val] ->
          let idx = Int32.to_int (to_int32 idx_val) in
          let old_val = arr.(idx) in
          arr.(idx) <- add_int32 old_val add_val ;
          old_val
      | _ -> failwith "atomic_add_global_int32: wrong arguments")
  (* Unknown intrinsic *)
  | _ ->
      let full_name = String.concat "." (path @ [name]) in
      failwith ("eval_intrinsic: unknown intrinsic " ^ full_name)

(** {1 Expression Evaluation} *)

let rec eval_expr state env expr =
  match expr with
  (* Literals *)
  | Int n -> VInt32 (Int32.of_int n)
  | Float f -> VFloat32 f
  | Double f -> VFloat64 f
  | Unit -> VUnit
  (* Variables *)
  | IntVar (id, _, _)
  | FloatVar (id, _, _)
  | DoubleVar (id, _, _)
  | BoolVar (id, _, _)
  | UnitVar (id, _, _) -> (
      try Hashtbl.find env.locals id
      with Not_found ->
        failwith ("eval_expr: unbound variable id " ^ string_of_int id))
  | Id name | IdName name -> (
      (* Try to find by name in arrays first *)
      try VArray (Hashtbl.find env.arrays name)
      with Not_found -> (
        try VArray (Hashtbl.find env.shared name)
        with Not_found -> failwith ("eval_expr: unbound id " ^ name)))
  | IntId (name, id) -> (
      try Hashtbl.find env.locals id
      with Not_found -> (
        (* Also check shared arrays - they're referenced by IntId in lowered AST *)
        try VArray (Hashtbl.find env.shared name)
        with Not_found ->
          failwith ("eval_expr: unbound IntId " ^ string_of_int id)))
  (* Global values *)
  | GInt f -> VInt32 (f ())
  | GFloat f -> VFloat32 (f ())
  | GFloat64 f -> VFloat64 (f ())
  (* Arithmetic - integer *)
  | Plus (a, b) -> add_int32 (eval_expr state env a) (eval_expr state env b)
  | Min (a, b) -> sub_int32 (eval_expr state env a) (eval_expr state env b)
  | Mul (a, b) -> mul_int32 (eval_expr state env a) (eval_expr state env b)
  | Div (a, b) -> div_int32 (eval_expr state env a) (eval_expr state env b)
  | Mod (a, b) -> mod_int32 (eval_expr state env a) (eval_expr state env b)
  (* Arithmetic - float *)
  | Plusf (a, b) -> add_float (eval_expr state env a) (eval_expr state env b)
  | Minf (a, b) -> sub_float (eval_expr state env a) (eval_expr state env b)
  | Mulf (a, b) -> mul_float (eval_expr state env a) (eval_expr state env b)
  | Divf (a, b) -> div_float (eval_expr state env a) (eval_expr state env b)
  (* Comparisons *)
  | EqBool (a, b) ->
      VBool (eq_val (eval_expr state env a) (eval_expr state env b))
  | LtBool (a, b) ->
      VBool (lt_val (eval_expr state env a) (eval_expr state env b))
  | GtBool (a, b) ->
      VBool (gt_val (eval_expr state env a) (eval_expr state env b))
  | LtEBool (a, b) ->
      VBool (lte_val (eval_expr state env a) (eval_expr state env b))
  | GtEBool (a, b) ->
      VBool (gte_val (eval_expr state env a) (eval_expr state env b))
  | EqCustom (_, a, b) ->
      VBool (eq_val (eval_expr state env a) (eval_expr state env b))
  (* Boolean ops / Bitwise ops - disambiguated by operand type *)
  | And (a, b) -> (
      let va = eval_expr state env a in
      let vb = eval_expr state env b in
      match (va, vb) with
      | VBool _, VBool _ -> VBool (to_bool va && to_bool vb)
      | _ -> VInt32 (Int32.logand (to_int32 va) (to_int32 vb)))
  | Or (a, b) -> (
      let va = eval_expr state env a in
      let vb = eval_expr state env b in
      match (va, vb) with
      | VBool _, VBool _ -> VBool (to_bool va || to_bool vb)
      | _ -> VInt32 (Int32.logor (to_int32 va) (to_int32 vb)))
  | Not a -> VBool (not (to_bool (eval_expr state env a)))
  (* Array access *)
  | Acc (arr_expr, idx_expr) | IntVecAcc (arr_expr, idx_expr) ->
      let arr = get_array state env arr_expr in
      let idx = to_int (eval_expr state env idx_expr) in
      if idx < 0 || idx >= Array.length arr then
        failwith
          (Printf.sprintf
             "Array index out of bounds: %d (length %d)"
             idx
             (Array.length arr))
      else arr.(idx)
  | VecVar (_, id, name) -> (
      try VArray (Hashtbl.find env.arrays name)
      with Not_found -> (
        try Hashtbl.find env.locals id
        with Not_found -> failwith ("eval_expr: unbound VecVar " ^ name)))
  (* Intrinsics *)
  | Intrinsics (_path, name) -> eval_intrinsic state [] name []
  | IntrinsicRef (path, name) -> eval_intrinsic state path name []
  (* Records and constructors *)
  | Record (name, fields) ->
      VRecord (name, Array.of_list (List.map (eval_expr state env) fields))
  | RecGet (rec_expr, _field_name) -> (
      match eval_expr state env rec_expr with
      | VRecord (_, fields) ->
          (* For now just return first field - proper impl needs field name mapping *)
          if Array.length fields > 0 then fields.(0)
          else failwith "RecGet: empty record"
      | _ -> failwith "RecGet: not a record")
  | Constr (type_name, _ctor_name, args) ->
      (* Tag is embedded in ctor_name or needs lookup - simplified here *)
      VConstr (type_name, 0, List.map (eval_expr state env) args)
  (* Conditionals *)
  | Ife (cond, then_e, else_e) ->
      if to_bool (eval_expr state env cond) then eval_expr state env then_e
      else eval_expr state env else_e
  | If (cond, then_e) ->
      if to_bool (eval_expr state env cond) then eval_expr state env then_e
      else VUnit
  (* Function application *)
  | App (fn_expr, args) -> (
      match fn_expr with
      | IntrinsicRef (path, name) ->
          let arg_vals = Array.to_list (Array.map (eval_expr state env) args) in
          eval_intrinsic state path name arg_vals
      | Intrinsics (_path, name) ->
          (* Intrinsics node used for some operations like spoc_xor *)
          let arg_vals = Array.to_list (Array.map (eval_expr state env) args) in
          eval_intrinsic state [] name arg_vals
      | GlobalFun (body, _name, _sig) ->
          (* Simple: evaluate body with args bound - needs proper impl *)
          eval_expr state env body
      | _ -> failwith "App: unsupported function expression")
  (* Pragmas - just evaluate body *)
  | Pragma (_, body) -> eval_expr state env body
  (* Native code blocks - try to interpret common patterns *)
  | Native f ->
      (* Call the native function with a fake interpreter device to get the code string *)
      let fake_dev =
        {
          Spoc.Devices.general_info =
            {
              Spoc.Devices.name = "Interpreter";
              totalGlobalMem = 0;
              localMemSize = 0;
              clockRate = 0;
              totalConstMem = 0;
              multiProcessorCount = 1;
              eccEnabled = false;
              id = 0;
              ctx = Obj.magic ();
            };
          specific_info =
            Spoc.Devices.InterpreterInfo
              {
                Spoc.Devices.backend = Spoc.Devices.Sequential;
                num_cores = 1;
                debug_mode = false;
              };
          gc_info = Obj.magic ();
          events = Obj.magic ();
        }
      in
      let code = f fake_dev in
      (* Try to interpret common atomic patterns *)
      if String.length code > 0 then begin
        (* atomicAdd (var,1) or atomicAdd(var, 1) - CUDA style *)
        let atomicAdd_re =
          Str.regexp "atomicAdd[ ]*(\\([a-zA-Z_][a-zA-Z0-9_]*\\),[ ]*1)"
        in
        (* atomic_inc (var) or atomic_inc(var) - OpenCL style *)
        let atomic_inc_re =
          Str.regexp "atomic_inc[ ]*(\\([a-zA-Z_][a-zA-Z0-9_]*\\))"
        in
        let matched_var =
          if Str.string_match atomicAdd_re code 0 then
            Some (Str.matched_group 1 code)
          else if Str.string_match atomic_inc_re code 0 then
            Some (Str.matched_group 1 code)
          else None
        in
        match matched_var with
        | Some var_name -> (
            try
              let arr = Hashtbl.find env.arrays var_name in
              match arr.(0) with
              | VInt32 n -> arr.(0) <- VInt32 (Int32.add n 1l)
              | VInt64 n -> arr.(0) <- VInt64 (Int64.add n 1L)
              | _ -> ()
            with Not_found -> ())
        | None -> ()
      end ;
      VUnit
  | NativeVar _ ->
      (* NativeVar references a native function by name - can't interpret *)
      VUnit
  (* Other *)
  | Empty -> VUnit
  | Return e -> eval_expr state env e
  (* Statement-like nodes that can appear in expression context due to let bindings *)
  | Seq (a, b) ->
      (* Execute first as statement-for-side-effect, then evaluate second *)
      let _ = eval_expr state env a in
      eval_expr state env b
  | Decl var_expr -> (
      (* Declaration - add to locals with default value *)
      match var_expr with
      | IntVar (id, _, _) ->
          Hashtbl.replace env.locals id (VInt32 0l) ;
          VUnit
      | FloatVar (id, _, _) ->
          Hashtbl.replace env.locals id (VFloat32 0.0) ;
          VUnit
      | DoubleVar (id, _, _) ->
          Hashtbl.replace env.locals id (VFloat64 0.0) ;
          VUnit
      | BoolVar (id, _, _) ->
          Hashtbl.replace env.locals id (VBool false) ;
          VUnit
      | UnitVar (id, _, _) ->
          Hashtbl.replace env.locals id VUnit ;
          VUnit
      | _ -> VUnit)
  | Set (var_expr, val_expr) -> (
      (* Assignment - set value in locals *)
      let value = eval_expr state env val_expr in
      match var_expr with
      | IntVar (id, _, _)
      | FloatVar (id, _, _)
      | DoubleVar (id, _, _)
      | BoolVar (id, _, _)
      | UnitVar (id, _, _) ->
          Hashtbl.replace env.locals id value ;
          VUnit
      | IntId (_, id) ->
          Hashtbl.replace env.locals id value ;
          VUnit
      | _ -> failwith "Set in eval_expr: unsupported lvalue")
  | SetV (Acc (arr_expr, idx_expr), val_expr)
  | SetV (IntVecAcc (arr_expr, idx_expr), val_expr) ->
      (* Array element assignment *)
      let arr = get_array state env arr_expr in
      let idx = to_int (eval_expr state env idx_expr) in
      let value = eval_expr state env val_expr in
      if idx >= 0 && idx < Array.length arr then arr.(idx) <- value ;
      VUnit
  | _ ->
      failwith
        ("eval_expr: unsupported expression " ^ Kirc_Ast.string_of_ast expr)

and get_array state env expr =
  match expr with
  | Id name | IdName name -> (
      try Hashtbl.find env.arrays name
      with Not_found -> (
        try Hashtbl.find env.shared name
        with Not_found -> failwith ("get_array: unknown array " ^ name)))
  | VecVar (_, _, name) -> (
      try Hashtbl.find env.arrays name
      with Not_found -> (
        try Hashtbl.find env.shared name
        with Not_found -> failwith ("get_array: unknown VecVar " ^ name)))
  | IntId (name, _) -> (
      (* Shared arrays are referenced by IntId in lowered AST *)
      try Hashtbl.find env.arrays name
      with Not_found -> (
        try Hashtbl.find env.shared name
        with Not_found -> failwith ("get_array: unknown IntId array " ^ name)))
  | _ -> (
      match eval_expr state env expr with
      | VArray arr -> arr
      | _ -> failwith "get_array: not an array")

(** {1 Statement Execution} *)

let rec exec_stmt state env stmt =
  match stmt with
  | Empty | Unit -> ()
  (* Sequence *)
  | Seq (a, b) ->
      exec_stmt state env a ;
      exec_stmt state env b
  | Concat (a, b) ->
      exec_stmt state env a ;
      exec_stmt state env b
  | Block body -> exec_stmt state env body
  (* Variable declaration *)
  | Decl var_expr -> (
      match var_expr with
      | IntVar (id, _, _) -> Hashtbl.replace env.locals id (VInt32 0l)
      | FloatVar (id, _, _) -> Hashtbl.replace env.locals id (VFloat32 0.0)
      | DoubleVar (id, _, _) -> Hashtbl.replace env.locals id (VFloat64 0.0)
      | BoolVar (id, _, _) -> Hashtbl.replace env.locals id (VBool false)
      | UnitVar (id, _, _) -> Hashtbl.replace env.locals id VUnit
      | _ -> ())
  (* Variable assignment *)
  | Set (var_expr, val_expr) -> (
      let value = eval_expr state env val_expr in
      match var_expr with
      | IntVar (id, _, _)
      | FloatVar (id, _, _)
      | DoubleVar (id, _, _)
      | BoolVar (id, _, _)
      | UnitVar (id, _, _) ->
          Hashtbl.replace env.locals id value
      | IntId (_, id) -> Hashtbl.replace env.locals id value
      | _ -> failwith "Set: unsupported lvalue")
  (* Array element assignment *)
  | SetV (Acc (arr_expr, idx_expr), val_expr)
  | SetV (IntVecAcc (arr_expr, idx_expr), val_expr) ->
      let arr = get_array state env arr_expr in
      let idx = to_int (eval_expr state env idx_expr) in
      let value = eval_expr state env val_expr in
      if idx < 0 || idx >= Array.length arr then
        failwith
          (Printf.sprintf
             "SetV: index out of bounds: %d (length %d)"
             idx
             (Array.length arr))
      else arr.(idx) <- value
  | SetV (lhs, _rhs) ->
      (* Generic SetV - unsupported lvalue *)
      failwith
        (Printf.sprintf
           "SetV: unsupported lvalue: %s"
           (Kirc_Ast.string_of_ast lhs))
  (* Local variable with init *)
  | SetLocalVar (var_expr, init_expr, body) -> (
      let value = eval_expr state env init_expr in
      match var_expr with
      | IntVar (id, _, _)
      | FloatVar (id, _, _)
      | DoubleVar (id, _, _)
      | BoolVar (id, _, _)
      | UnitVar (id, _, _) ->
          Hashtbl.replace env.locals id value ;
          exec_stmt state env body
      | _ -> failwith "SetLocalVar: unsupported variable")
  (* Local array allocation *)
  | Local (Arr (name, size_expr, elt_type, memspace), body) -> (
      let size = to_int (eval_expr state env size_expr) in
      let init_val =
        match elt_type with
        | EInt32 -> VInt32 0l
        | EInt64 -> VInt64 0L
        | EFloat32 -> VFloat32 0.0
        | EFloat64 -> VFloat64 0.0
      in
      (* For shared memory, reuse existing array if already allocated.
         This is important for BSP execution where multiple threads share
         the same shared memory array within a block. Shared memory is
         cleared at the start of each block by run_grid, so we don't
         remove it here - it persists for the block's lifetime. *)
      (match memspace with
      | Shared -> (
          match Hashtbl.find_opt env.shared name with
          | Some _ -> () (* Already allocated by another thread *)
          | None ->
              let arr = Array.make size init_val in
              Hashtbl.add env.shared name arr)
      | _ ->
          let arr = Array.make size init_val in
          Hashtbl.add env.arrays name arr) ;
      exec_stmt state env body ;
      (* Only remove private arrays, not shared (shared cleared per-block) *)
      match memspace with
      | Shared -> ()
      | _ -> Hashtbl.remove env.arrays name)
  | Local (decl, body) ->
      exec_stmt state env (Decl decl) ;
      exec_stmt state env body
  (* Conditionals *)
  | Ife (cond, then_s, else_s) ->
      if to_bool (eval_expr state env cond) then exec_stmt state env then_s
      else exec_stmt state env else_s
  | If (cond, then_s) ->
      if to_bool (eval_expr state env cond) then exec_stmt state env then_s
  (* Loops *)
  | DoLoop (var_expr, start_expr, stop_expr, body) -> (
      let start_val = to_int32 (eval_expr state env start_expr) in
      let stop_val = to_int32 (eval_expr state env stop_expr) in
      match var_expr with
      | IntVar (id, _, _) | IntId (_, id) ->
          let i = ref start_val in
          while !i <= stop_val do
            Hashtbl.replace env.locals id (VInt32 !i) ;
            exec_stmt state env body ;
            i := Int32.add !i 1l
          done
      | _ -> failwith "DoLoop: unsupported loop variable")
  | While (cond, body) ->
      while to_bool (eval_expr state env cond) do
        exec_stmt state env body
      done
  (* Match/case *)
  | Match (_, scrutinee, cases) ->
      let value = eval_expr state env scrutinee in
      let tag =
        match value with
        | VConstr (_, tag, _) -> tag
        | VInt32 n -> Int32.to_int n
        | _ -> 0
      in
      let _, _, body =
        let rec find_case i =
          if i >= Array.length cases then cases.(0)
          else
            let ((t, _, _) as case) = cases.(i) in
            if t = tag then case else find_case (i + 1)
        in
        find_case 0
      in
      exec_stmt state env body
  (* Record field set *)
  | RecSet (rec_expr, val_expr) -> (
      match eval_expr state env rec_expr with
      | VRecord (_, _fields) ->
          let _value = eval_expr state env val_expr in
          () (* Would need field index to update *)
      | _ -> failwith "RecSet: not a record")
  (* Return - just evaluate for side effects *)
  | Return e ->
      let _ = eval_expr state env e in
      ()
  (* Barriers - perform effect to synchronize with other threads *)
  | IntrinsicRef ((["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "block_barrier") ->
      Effect.perform Barrier
  | IntrinsicRef ((["Gpu"] | ["Sarek_stdlib"; "Gpu"]), "warp_barrier") ->
      Effect.perform Barrier
  (* Pragmas *)
  | Pragma (_, body) -> exec_stmt state env body
  (* Global function definition - store for later *)
  | GlobalFun (body, name, _sig) -> Hashtbl.add env.functions name body
  (* Kernel wrapper *)
  | Kern (params, body) ->
      exec_stmt state env params ;
      exec_stmt state env body
  | Params p -> exec_stmt state env p
  (* Fallback to expression evaluation *)
  | other ->
      let _ = eval_expr state env other in
      ()

(** {1 Kernel Execution} *)

(** Run a single thread *)
let run_thread state env body = exec_stmt state env body

(** Run all threads in a block with BSP barrier synchronization. Uses effects to
    suspend threads at barriers and resume them together. *)
let run_block env body block_idx block_dim grid_dim =
  let bx, by, bz = block_dim in
  let num_threads = bx * by * bz in

  (* Continuations waiting at barrier *)
  let waiting : (unit, unit) Effect.Deep.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in

  (* Run a single thread with effect handler *)
  let run_thread_with_barrier tid =
    let tx = tid mod bx in
    let ty = tid / bx mod by in
    let tz = tid / (bx * by) in
    let state = {thread_idx = (tx, ty, tz); block_idx; block_dim; grid_dim} in
    let thread_env = copy_env env in
    Effect.Deep.match_with
      (fun () -> run_thread state thread_env body)
      ()
      {
        retc = (fun () -> incr num_completed);
        exnc = raise;
        effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Barrier ->
                Some
                  (fun (k : (a, unit) Effect.Deep.continuation) ->
                    waiting.(tid) <- Some k ;
                    incr num_waiting)
            | _ -> None);
      }
  in

  (* Resume a waiting thread *)
  let resume_thread tid =
    match waiting.(tid) with
    | Some k ->
        waiting.(tid) <- None ;
        Effect.Deep.match_with
          (fun () -> Effect.Deep.continue k ())
          ()
          {
            retc = (fun () -> incr num_completed);
            exnc = raise;
            effc =
              (fun (type a) (eff : a Effect.t) ->
                match eff with
                | Barrier ->
                    Some
                      (fun (k : (a, unit) Effect.Deep.continuation) ->
                        waiting.(tid) <- Some k ;
                        incr num_waiting)
                | _ -> None);
          }
    | None -> ()
  in

  (* Initial run: start all threads *)
  for tid = 0 to num_threads - 1 do
    run_thread_with_barrier tid
  done ;

  (* Superstep loop: while threads are waiting at barriers *)
  while !num_waiting > 0 do
    let to_resume = !num_waiting in
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if Option.is_some waiting.(tid) then resume_thread tid
    done ;
    if !num_waiting = to_resume && !num_completed < num_threads then
      failwith "BSP deadlock: no progress made in interpreter"
  done

(** Run all blocks in a grid *)
let run_grid env body block_dim grid_dim =
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        (* Clear shared memory for each block *)
        Hashtbl.clear env.shared ;
        run_block env body (bx, by, bz) block_dim grid_dim
      done
    done
  done

(** {1 Public API} *)

(** Extract kernel body from k_ext *)
let extract_body = function Kern (_, body) -> body | body -> body

(** Parameter info: name, is_scalar, and optional var_id for scalars *)
type param_info = {
  p_name : string;
  p_is_scalar : bool;
  p_var_id : int option;  (** Variable ID for scalar params *)
}

(** Argument type for run_body_with_params *)
type arg_value =
  | ArgArray of value array  (** Vector parameter *)
  | ArgScalar of value  (** Scalar parameter *)

(** Extract parameter names from Kern(params, body) in order. Returns list of
    (name, is_scalar) pairs for backward compatibility. *)
let rec extract_param_names = function
  | Kern (params, _) ->
      List.map (fun p -> (p.p_name, p.p_is_scalar)) (collect_params params)
  | _ -> []

(** Extract full parameter info from Kern(params, body). *)
and extract_param_info = function
  | Kern (params, _) -> collect_params params
  | _ -> []

and collect_params = function
  | Params p -> collect_params p
  | Concat (a, b) -> collect_params a @ collect_params b
  | VecVar (_, id, name) ->
      [{p_name = name; p_is_scalar = false; p_var_id = Some id}]
  | IntVar (id, name, _) ->
      [{p_name = name; p_is_scalar = true; p_var_id = Some id}]
  | FloatVar (id, name, _) ->
      [{p_name = name; p_is_scalar = true; p_var_id = Some id}]
  | DoubleVar (id, name, _) ->
      [{p_name = name; p_is_scalar = true; p_var_id = Some id}]
  | Arr (name, _, _, _) ->
      [{p_name = name; p_is_scalar = false; p_var_id = None}]
  | _ -> []

(** Run a kernel body on CPU with full parameter info *)
let run_body_with_params body ~block ~grid
    (args : (param_info * arg_value) list) =
  let env = create_env () in
  List.iter
    (fun (pinfo, arg) ->
      match arg with
      | ArgArray arr -> (
          (* Vector: store by name in arrays, and by var_id in locals as VArray *)
          Hashtbl.add env.arrays pinfo.p_name arr ;
          match pinfo.p_var_id with
          | Some id -> Hashtbl.add env.locals id (VArray arr)
          | None -> ())
      | ArgScalar v -> (
          (* Scalar: store both by name (as single-element array) and by var_id *)
          Hashtbl.add env.arrays pinfo.p_name [|v|] ;
          match pinfo.p_var_id with
          | Some id -> Hashtbl.add env.locals id v
          | None -> ()))
    args ;
  run_grid env body block grid

(** Run a kernel body on CPU (legacy interface) *)
let run_body body ~block ~grid (arrays : (string * value array) list) =
  let env = create_env () in
  List.iter (fun (name, arr) -> Hashtbl.add env.arrays name arr) arrays ;
  run_grid env body block grid

(** Convenience: create int32 array *)
let int32_array n = Array.make n (VInt32 0l)

(** Convenience: create float32 array *)
let float32_array n = Array.make n (VFloat32 0.0)

(** Get int32 values from array *)
let get_int32s arr = Array.map to_int32 arr

(** Get float32 values from array *)
let get_float32s arr = Array.map to_float32 arr

(** Set int32 values in array *)
let set_int32s arr values = Array.iteri (fun i v -> arr.(i) <- VInt32 v) values

(** Set float32 values in array *)
let set_float32s arr values =
  Array.iteri (fun i v -> arr.(i) <- VFloat32 v) values

(** Wrap a SPOC vector into interpreter value array. This creates a value array
    that references the same underlying data. *)
let wrap_vector (vec : ('a, 'b) Spoc.Vector.vector) : value array =
  let len = Spoc.Vector.length vec in
  let arr = Array.make len VUnit in
  (* Determine element type from vector kind *)
  (match Spoc.Vector.kind vec with
  | Spoc.Vector.Float32 _ ->
      for i = 0 to len - 1 do
        arr.(i) <- VFloat32 (Obj.magic (Spoc.Mem.get vec i) : float)
      done
  | Spoc.Vector.Float64 _ ->
      for i = 0 to len - 1 do
        arr.(i) <- VFloat64 (Obj.magic (Spoc.Mem.get vec i) : float)
      done
  | Spoc.Vector.Int32 _ ->
      for i = 0 to len - 1 do
        arr.(i) <- VInt32 (Obj.magic (Spoc.Mem.get vec i) : int32)
      done
  | Spoc.Vector.Int64 _ ->
      for i = 0 to len - 1 do
        arr.(i) <- VInt64 (Obj.magic (Spoc.Mem.get vec i) : int64)
      done
  | Spoc.Vector.Char _ ->
      for i = 0 to len - 1 do
        arr.(i) <-
          VInt32
            (Int32.of_int (Char.code (Obj.magic (Spoc.Mem.get vec i) : char)))
      done
  | _ -> failwith "wrap_vector: unsupported vector type") ;
  arr

(** Unwrap interpreter value array back to SPOC vector. This copies the values
    back to the vector. *)
let unwrap_to_vector arr (vec : ('a, 'b) Spoc.Vector.vector) : unit =
  let len = min (Array.length arr) (Spoc.Vector.length vec) in
  match Spoc.Vector.kind vec with
  | Spoc.Vector.Float32 _ ->
      for i = 0 to len - 1 do
        Spoc.Mem.set vec i (Obj.magic (to_float32 arr.(i)))
      done
  | Spoc.Vector.Float64 _ ->
      for i = 0 to len - 1 do
        Spoc.Mem.set vec i (Obj.magic (to_float64 arr.(i)))
      done
  | Spoc.Vector.Int32 _ ->
      for i = 0 to len - 1 do
        Spoc.Mem.set vec i (Obj.magic (to_int32 arr.(i)))
      done
  | Spoc.Vector.Int64 _ ->
      for i = 0 to len - 1 do
        Spoc.Mem.set vec i (Obj.magic (to_int64 arr.(i)))
      done
  | _ -> failwith "unwrap_to_vector: unsupported vector type"
