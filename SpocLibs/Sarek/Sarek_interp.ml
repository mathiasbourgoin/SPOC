(** Sarek_interp - CPU interpreter for Sarek kernels

    Executes Kirc_Ast on CPU for debugging and testing without GPU. *)

open Kirc_Ast

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

let lt_val a b = to_int32 a < to_int32 b

let gt_val a b = to_int32 a > to_int32 b

let lte_val a b = to_int32 a <= to_int32 b

let gte_val a b = to_int32 a >= to_int32 b

(** {1 Intrinsics} *)

let eval_intrinsic state path name args =
  match (path, name) with
  (* Thread indices *)
  | (["Gpu"] | []), "thread_idx_x" ->
      let x, _, _ = state.thread_idx in
      VInt32 (Int32.of_int x)
  | (["Gpu"] | []), "thread_idx_y" ->
      let _, y, _ = state.thread_idx in
      VInt32 (Int32.of_int y)
  | (["Gpu"] | []), "thread_idx_z" ->
      let _, _, z = state.thread_idx in
      VInt32 (Int32.of_int z)
  (* Block indices *)
  | (["Gpu"] | []), "block_idx_x" ->
      let x, _, _ = state.block_idx in
      VInt32 (Int32.of_int x)
  | (["Gpu"] | []), "block_idx_y" ->
      let _, y, _ = state.block_idx in
      VInt32 (Int32.of_int y)
  | (["Gpu"] | []), "block_idx_z" ->
      let _, _, z = state.block_idx in
      VInt32 (Int32.of_int z)
  (* Block dimensions *)
  | (["Gpu"] | []), "block_dim_x" ->
      let x, _, _ = state.block_dim in
      VInt32 (Int32.of_int x)
  | (["Gpu"] | []), "block_dim_y" ->
      let _, y, _ = state.block_dim in
      VInt32 (Int32.of_int y)
  | (["Gpu"] | []), "block_dim_z" ->
      let _, _, z = state.block_dim in
      VInt32 (Int32.of_int z)
  (* Grid dimensions *)
  | (["Gpu"] | []), "grid_dim_x" ->
      let x, _, _ = state.grid_dim in
      VInt32 (Int32.of_int x)
  | (["Gpu"] | []), "grid_dim_y" ->
      let _, y, _ = state.grid_dim in
      VInt32 (Int32.of_int y)
  | (["Gpu"] | []), "grid_dim_z" ->
      let _, _, z = state.grid_dim in
      VInt32 (Int32.of_int z)
  (* Global index helpers *)
  | (["Gpu"] | []), "global_idx" ->
      let tx, _, _ = state.thread_idx in
      let bx, _, _ = state.block_idx in
      let bdx, _, _ = state.block_dim in
      VInt32 (Int32.of_int ((bx * bdx) + tx))
  | (["Gpu"] | []), "global_size" ->
      let bdx, _, _ = state.block_dim in
      let gdx, _, _ = state.grid_dim in
      VInt32 (Int32.of_int (bdx * gdx))
  (* Barriers - no-op in sequential mode *)
  | (["Gpu"] | []), "block_barrier" -> VUnit
  | (["Gpu"] | []), "warp_barrier" -> VUnit
  (* Math intrinsics - Float32 *)
  | ["Float32"], "sin" -> VFloat32 (sin (to_float32 (List.hd args)))
  | ["Float32"], "cos" -> VFloat32 (cos (to_float32 (List.hd args)))
  | ["Float32"], "tan" -> VFloat32 (tan (to_float32 (List.hd args)))
  | ["Float32"], "sqrt" -> VFloat32 (sqrt (to_float32 (List.hd args)))
  | ["Float32"], "exp" -> VFloat32 (exp (to_float32 (List.hd args)))
  | ["Float32"], "log" -> VFloat32 (log (to_float32 (List.hd args)))
  | ["Float32"], "abs" -> VFloat32 (abs_float (to_float32 (List.hd args)))
  | ["Float32"], "floor" -> VFloat32 (floor (to_float32 (List.hd args)))
  | ["Float32"], "ceil" -> VFloat32 (ceil (to_float32 (List.hd args)))
  | ["Float32"], "fma" ->
      let a = to_float32 (List.nth args 0) in
      let b = to_float32 (List.nth args 1) in
      let c = to_float32 (List.nth args 2) in
      VFloat32 ((a *. b) +. c)
  | ["Float32"], "min" ->
      VFloat32
        (min (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | ["Float32"], "max" ->
      VFloat32
        (max (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  (* Int32 math *)
  | ["Int32"], "abs" -> VInt32 (Int32.abs (to_int32 (List.hd args)))
  | ["Int32"], "min" ->
      VInt32 (min (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | ["Int32"], "max" ->
      VInt32 (max (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  (* Type conversions *)
  | (["Std"] | ["Gpu"] | []), "float" ->
      VFloat32 (Int32.to_float (to_int32 (List.hd args)))
  | (["Std"] | ["Gpu"] | []), "float64" ->
      VFloat64 (Int32.to_float (to_int32 (List.hd args)))
  | (["Std"] | ["Gpu"] | []), "int_of_float" ->
      VInt32 (Int32.of_float (to_float32 (List.hd args)))
  | (["Std"] | ["Gpu"] | []), "int_of_float64" ->
      VInt32 (Int32.of_float (to_float64 (List.hd args)))
  | ["Float32"], "of_int" -> VFloat32 (Int32.to_float (to_int32 (List.hd args)))
  | ["Float64"], "of_int" -> VFloat64 (Int32.to_float (to_int32 (List.hd args)))
  | ["Float64"], "of_float32" -> VFloat64 (to_float32 (List.hd args))
  | ["Float32"], "of_float64" -> VFloat32 (to_float64 (List.hd args))
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
  | IntId (_, id) -> (
      try Hashtbl.find env.locals id
      with Not_found ->
        failwith ("eval_expr: unbound IntId " ^ string_of_int id))
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
  (* Boolean ops *)
  | And (a, b) ->
      VBool (to_bool (eval_expr state env a) && to_bool (eval_expr state env b))
  | Or (a, b) ->
      VBool (to_bool (eval_expr state env a) || to_bool (eval_expr state env b))
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
      | GlobalFun (body, _name, _sig) ->
          (* Simple: evaluate body with args bound - needs proper impl *)
          eval_expr state env body
      | _ -> failwith "App: unsupported function expression")
  (* Pragmas - just evaluate body *)
  | Pragma (_, body) -> eval_expr state env body
  (* Other *)
  | Empty -> VUnit
  | Return e -> eval_expr state env e
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
      let arr = Array.make size init_val in
      (match memspace with
      | Shared -> Hashtbl.add env.shared name arr
      | _ -> Hashtbl.add env.arrays name arr) ;
      exec_stmt state env body ;
      match memspace with
      | Shared -> Hashtbl.remove env.shared name
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
  (* Barriers - no-op in sequential mode *)
  | IntrinsicRef (["Gpu"], "block_barrier") -> ()
  | IntrinsicRef (["Gpu"], "warp_barrier") -> ()
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

(** Run all threads in a block (sequential mode) *)
let run_block env body block_idx block_dim grid_dim =
  let bx, by, bz = block_dim in
  for tz = 0 to bz - 1 do
    for ty = 0 to by - 1 do
      for tx = 0 to bx - 1 do
        let state =
          {thread_idx = (tx, ty, tz); block_idx; block_dim; grid_dim}
        in
        let thread_env = copy_env env in
        run_thread state thread_env body
      done
    done
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

(** Run a kernel body on CPU *)
let run_body body ~block ~grid (arrays : (string * value array) list) =
  let env = create_env () in
  List.iter (fun (name, arr) -> Hashtbl.add env.arrays name arr) arrays ;
  run_grid env body block grid

(** Extract kernel body from k_ext *)
let extract_body = function Kern (_, body) -> body | body -> body

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
