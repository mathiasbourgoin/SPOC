(******************************************************************************
 * Sarek_ir_interp - CPU interpreter for Sarek V2 IR kernels
 *
 * Executes Sarek_ir.kernel on CPU for debugging and testing without GPU.
 * Supports BSP-style barrier synchronization using OCaml 5 effects.
 *
 * This is the V2 counterpart to Sarek_interp (which works on Kirc_Ast).
 ******************************************************************************)

open Sarek_ir
module F32 = Sarek_float32

(** Re-export value type from Sarek_value for convenience *)
type value = Sarek_value.value =
  | VInt32 of int32
  | VInt64 of int64
  | VFloat32 of float
  | VFloat64 of float
  | VBool of bool
  | VUnit
  | VArray of value array
  | VRecord of string * value array
  | VVariant of string * int * value list

(** {1 BSP Barrier Effect}
    Used for synchronizing threads at barriers. Each thread is suspended when it
    hits a barrier, and all threads are resumed together. *)

type _ Effect.t += Barrier : unit Effect.t

(** {1 Thread State} *)

type thread_state = {
  thread_idx : int * int * int;
  block_idx : int * int * int;
  block_dim : int * int * int;
  grid_dim : int * int * int;
}

(** {1 Environment} *)

type env = {
  vars : (int, value) Hashtbl.t;  (** var_id -> value *)
  vars_by_name : (string, value) Hashtbl.t;  (** var_name -> value (fallback) *)
  arrays : (string, value array) Hashtbl.t;  (** array_name -> data *)
  shared : (string, value array) Hashtbl.t;  (** shared arrays for block *)
  funcs : (string, helper_func) Hashtbl.t;  (** helper functions *)
}

let create_env () =
  {
    vars = Hashtbl.create 32;
    vars_by_name = Hashtbl.create 32;
    arrays = Hashtbl.create 16;
    shared = Hashtbl.create 8;
    funcs = Hashtbl.create 8;
  }

let copy_env env =
  {
    vars = Hashtbl.copy env.vars;
    vars_by_name = Hashtbl.copy env.vars_by_name;
    arrays = env.arrays;
    (* shared across threads *)
    shared = env.shared;
    (* shared within block *)
    funcs = env.funcs;
    (* shared *)
  }

(** Bind a variable in the environment (both by id and name) *)
let bind_var env (v : var) value =
  Hashtbl.replace env.vars v.var_id value ;
  Hashtbl.replace env.vars_by_name v.var_name value

(** Look up a variable (try id first, then name as fallback) *)
let lookup_var env (v : var) =
  match Hashtbl.find_opt env.vars v.var_id with
  | Some value -> value
  | None -> (
      match Hashtbl.find_opt env.vars_by_name v.var_name with
      | Some value -> value
      | None ->
          Interp_error.raise_error
            (Unbound_variable {name = v.var_name; context = "eval_expr"}))

(** {1 Value Operations} *)

let to_int32 = function
  | VInt32 n -> n
  | VInt64 n -> Int64.to_int32 n
  | VFloat32 f -> Int32.of_float f
  | VFloat64 f -> Int32.of_float f
  | VBool b -> if b then 1l else 0l
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int32";
             context = "to_int32";
           })

let to_int64 = function
  | VInt64 n -> n
  | VInt32 n -> Int64.of_int32 n
  | VFloat32 f -> Int64.of_float f
  | VFloat64 f -> Int64.of_float f
  | VBool b -> if b then 1L else 0L
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "int64";
             context = "to_int64";
           })

let to_int v = Int32.to_int (to_int32 v)

let to_float32 = function
  | VFloat32 f -> f
  | VFloat64 f -> F32.to_float32 f
  | VInt32 n -> F32.to_float32 (Int32.to_float n)
  | VInt64 n -> F32.to_float32 (Int64.to_float n)
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float32";
             context = "to_float32";
           })

let to_float64 = function
  | VFloat64 f -> f
  | VFloat32 f -> f
  | VInt32 n -> Int32.to_float n
  | VInt64 n -> Int64.to_float n
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "float64";
             context = "to_float64";
           })

let to_bool = function
  | VBool b -> b
  | VInt32 n -> n <> 0l
  | VInt64 n -> n <> 0L
  | VFloat32 f -> f <> 0.0
  | VFloat64 f -> f <> 0.0
  | v ->
      Interp_error.raise_error
        (Type_conversion_error
           {
             from_type = Sarek_value.value_type_name v;
             to_type = "bool";
             context = "to_bool";
           })

(** {1 Binary Operations} *)

let eval_binop op v1 v2 =
  match op with
  | Add -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.add a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a +. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.add (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 +. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.add (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.add (to_int32 v1) (to_int32 v2)))
  | Sub -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.sub a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a -. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.sub (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 -. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.sub (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.sub (to_int32 v1) (to_int32 v2)))
  | Mul -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.mul a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a *. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.mul (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 *. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.mul (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.mul (to_int32 v1) (to_int32 v2)))
  | Div -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VFloat32 (F32.div a b)
      | VFloat64 a, VFloat64 b -> VFloat64 (a /. b)
      | VFloat32 _, _ | _, VFloat32 _ ->
          VFloat32 (F32.div (to_float32 v1) (to_float32 v2))
      | VFloat64 _, _ | _, VFloat64 _ ->
          VFloat64 (to_float64 v1 /. to_float64 v2)
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.div (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.div (to_int32 v1) (to_int32 v2)))
  | Mod -> (
      match (v1, v2) with
      | VInt64 _, _ | _, VInt64 _ ->
          VInt64 (Int64.rem (to_int64 v1) (to_int64 v2))
      | _ -> VInt32 (Int32.rem (to_int32 v1) (to_int32 v2)))
  | Eq -> VBool (v1 = v2)
  | Ne -> VBool (v1 <> v2)
  | Lt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a < b)
      | VFloat64 a, VFloat64 b -> VBool (a < b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 < to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 < to_float64 v2)
      | _ -> VBool (to_int32 v1 < to_int32 v2))
  | Le -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a <= b)
      | VFloat64 a, VFloat64 b -> VBool (a <= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 <= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 <= to_float64 v2)
      | _ -> VBool (to_int32 v1 <= to_int32 v2))
  | Gt -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a > b)
      | VFloat64 a, VFloat64 b -> VBool (a > b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 > to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 > to_float64 v2)
      | _ -> VBool (to_int32 v1 > to_int32 v2))
  | Ge -> (
      match (v1, v2) with
      | VFloat32 a, VFloat32 b -> VBool (a >= b)
      | VFloat64 a, VFloat64 b -> VBool (a >= b)
      | VFloat32 _, _ | _, VFloat32 _ -> VBool (to_float32 v1 >= to_float32 v2)
      | VFloat64 _, _ | _, VFloat64 _ -> VBool (to_float64 v1 >= to_float64 v2)
      | _ -> VBool (to_int32 v1 >= to_int32 v2))
  | And -> VBool (to_bool v1 && to_bool v2)
  | Or -> VBool (to_bool v1 || to_bool v2)
  | Shl -> VInt32 (Int32.shift_left (to_int32 v1) (to_int v2))
  | Shr -> VInt32 (Int32.shift_right_logical (to_int32 v1) (to_int v2))
  | BitAnd -> VInt32 (Int32.logand (to_int32 v1) (to_int32 v2))
  | BitOr -> VInt32 (Int32.logor (to_int32 v1) (to_int32 v2))
  | BitXor -> VInt32 (Int32.logxor (to_int32 v1) (to_int32 v2))

let eval_unop op v =
  match op with
  | Neg -> (
      match v with
      | VFloat32 f -> VFloat32 (-.f)
      | VFloat64 f -> VFloat64 (-.f)
      | VInt64 n -> VInt64 (Int64.neg n)
      | _ -> VInt32 (Int32.neg (to_int32 v)))
  | Not -> VBool (not (to_bool v))
  | BitNot -> VInt32 (Int32.lognot (to_int32 v))

(** {1 Intrinsics} *)

let is_gpu_path = function
  | ["Gpu"] | [] | ["Std"] | ["Sarek_stdlib"; "Gpu"] | ["Sarek_stdlib"; "Std"]
    ->
      true
  | _ -> false

let is_float32_path = function
  | ["Float32"] | ["Sarek_stdlib"; "Float32"] -> true
  | _ -> false

let is_float64_path = function
  | ["Float64"] | ["Sarek_stdlib"; "Float64"] -> true
  | _ -> false

let is_int32_path = function
  | ["Int32"] | ["Sarek_stdlib"; "Int32"] -> true
  | _ -> false

let rec eval_intrinsic state path name args =
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
  | path, "global_idx_x" when is_gpu_path path ->
      let tx, _, _ = state.thread_idx in
      let bx, _, _ = state.block_idx in
      let bdx, _, _ = state.block_dim in
      VInt32 (Int32.of_int ((bx * bdx) + tx))
  | path, "global_thread_id" when is_gpu_path path ->
      let tx, _, _ = state.thread_idx in
      let bx, _, _ = state.block_idx in
      let bdx, _, _ = state.block_dim in
      VInt32 (Int32.of_int ((bx * bdx) + tx))
  | path, "global_idx_y" when is_gpu_path path ->
      let _, ty, _ = state.thread_idx in
      let _, by, _ = state.block_idx in
      let _, bdy, _ = state.block_dim in
      VInt32 (Int32.of_int ((by * bdy) + ty))
  | path, "global_idx_z" when is_gpu_path path ->
      let _, _, tz = state.thread_idx in
      let _, _, bz = state.block_idx in
      let _, _, bdz = state.block_dim in
      VInt32 (Int32.of_int ((bz * bdz) + tz))
  | path, "global_size_y" when is_gpu_path path ->
      let _, bdy, _ = state.block_dim in
      let _, gdy, _ = state.grid_dim in
      VInt32 (Int32.of_int (bdy * gdy))
  | path, "global_size_z" when is_gpu_path path ->
      let _, _, bdz = state.block_dim in
      let _, _, gdz = state.grid_dim in
      VInt32 (Int32.of_int (bdz * gdz))
  | path, "global_size" when is_gpu_path path ->
      let bdx, _, _ = state.block_dim in
      let gdx, _, _ = state.grid_dim in
      VInt32 (Int32.of_int (bdx * gdx))
  | path, "global_size_x" when is_gpu_path path ->
      let bdx, _, _ = state.block_dim in
      let gdx, _, _ = state.grid_dim in
      VInt32 (Int32.of_int (bdx * gdx))
  (* Barriers *)
  | path, "block_barrier" when is_gpu_path path ->
      Effect.perform Barrier ;
      VUnit
  | path, "warp_barrier" when is_gpu_path path ->
      Effect.perform Barrier ;
      VUnit
  (* Float32 intrinsics *)
  | path, "sin" when is_float32_path path ->
      VFloat32 (F32.sin (to_float32 (List.hd args)))
  | path, "cos" when is_float32_path path ->
      VFloat32 (F32.cos (to_float32 (List.hd args)))
  | path, "tan" when is_float32_path path ->
      VFloat32 (F32.tan (to_float32 (List.hd args)))
  | path, "sqrt" when is_float32_path path ->
      VFloat32 (F32.sqrt (to_float32 (List.hd args)))
  | path, "exp" when is_float32_path path ->
      VFloat32 (F32.exp (to_float32 (List.hd args)))
  | path, "log" when is_float32_path path ->
      VFloat32 (F32.log (to_float32 (List.hd args)))
  | path, "abs" when is_float32_path path ->
      VFloat32 (F32.abs (to_float32 (List.hd args)))
  | path, "floor" when is_float32_path path ->
      VFloat32 (F32.floor (to_float32 (List.hd args)))
  | path, "ceil" when is_float32_path path ->
      VFloat32 (F32.ceil (to_float32 (List.hd args)))
  | path, "pow" when is_float32_path path ->
      VFloat32
        (F32.pow (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | path, "min" when is_float32_path path ->
      VFloat32
        (F32.min (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | path, "max" when is_float32_path path ->
      VFloat32
        (F32.max (to_float32 (List.nth args 0)) (to_float32 (List.nth args 1)))
  | path, "of_int" when is_float32_path path ->
      VFloat32 (F32.of_int (to_int (List.hd args)))
  (* Float64 intrinsics *)
  | path, "sin" when is_float64_path path ->
      VFloat64 (sin (to_float64 (List.hd args)))
  | path, "cos" when is_float64_path path ->
      VFloat64 (cos (to_float64 (List.hd args)))
  | path, "sqrt" when is_float64_path path ->
      VFloat64 (sqrt (to_float64 (List.hd args)))
  | path, "exp" when is_float64_path path ->
      VFloat64 (exp (to_float64 (List.hd args)))
  | path, "log" when is_float64_path path ->
      VFloat64 (log (to_float64 (List.hd args)))
  | path, "abs" when is_float64_path path ->
      VFloat64 (Float.abs (to_float64 (List.hd args)))
  | path, "of_int" when is_float64_path path ->
      VFloat64 (Float.of_int (to_int (List.hd args)))
  (* Int32 intrinsics *)
  | path, "abs" when is_int32_path path ->
      VInt32 (Int32.abs (to_int32 (List.hd args)))
  | path, "min" when is_int32_path path ->
      VInt32 (min (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  | path, "max" when is_int32_path path ->
      VInt32 (max (to_int32 (List.nth args 0)) (to_int32 (List.nth args 1)))
  (* Type conversions *)
  | path, "float" when is_gpu_path path ->
      VFloat32 (F32.of_int (to_int (List.hd args)))
  | path, "float64" when is_gpu_path path ->
      VFloat64 (Float.of_int (to_int (List.hd args)))
  | path, "int_of_float" when is_gpu_path path ->
      VInt32 (Int32.of_float (to_float32 (List.hd args)))
  | path, "int_of_float64" when is_gpu_path path ->
      VInt32 (Int32.of_float (to_float64 (List.hd args)))
  (* Unknown *)
  | _ ->
      let full = String.concat "." (path @ [name]) in
      Interp_error.raise_error (Unknown_intrinsic {name = full})

(** {1 Expression Evaluation} *)

(** Array expression evaluation *)
and eval_array_expr state env = function
  | EArrayRead (arr, idx) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx) in
      if i < 0 || i >= Array.length a then
        Interp_error.raise_error
          (Array_bounds_error
             {array_name = arr; index = i; length = Array.length a})
      else a.(i)
  | EArrayReadExpr (base, idx) ->
      let a =
        match eval_expr state env base with
        | VArray arr -> arr
        | _ ->
            Interp_error.raise_error
              (Not_an_array {expr = "EArrayReadExpr base"})
      in
      let i = to_int (eval_expr state env idx) in
      a.(i)
  | EArrayLen arr ->
      let a = get_array env arr in
      VInt32 (Int32.of_int (Array.length a))
  | EArrayCreate (ty, size_expr, _memspace) ->
      let size = to_int (eval_expr state env size_expr) in
      let init =
        match ty with
        | TInt32 -> VInt32 0l
        | TInt64 -> VInt64 0L
        | TFloat32 -> VFloat32 0.0
        | TFloat64 -> VFloat64 0.0
        | TBool -> VBool false
        | _ -> VUnit
      in
      VArray (Array.make size init)
  | _ -> assert false

(** Record and variant expression evaluation *)
and eval_composite_expr state env = function
  | ERecordField (e, field) -> (
      match eval_expr state env e with
      | VRecord (type_name, fields) as vrec -> (
          match Sarek_type_helpers.lookup type_name with
          | Some h ->
              let native_record = h.from_value vrec in
              h.get_field native_record field
          | None ->
              let field_infos = Sarek_registry.record_fields type_name in
              let rec find_idx i = function
                | [] ->
                    Interp_error.raise_error
                      (Pattern_match_failure
                         {
                           context =
                             Printf.sprintf
                               "Record field %s not found in %s"
                               field
                               type_name;
                         })
                | info :: rest ->
                    if info.Sarek_registry.field_name = field then i
                    else find_idx (i + 1) rest
              in
              let idx = find_idx 0 field_infos in
              fields.(idx))
      | _ -> Interp_error.raise_error (Not_a_record {expr = "ERecordField"}))
  | ERecord (name, fields) ->
      VRecord
        ( name,
          Array.of_list (List.map (fun (_, e) -> eval_expr state env e) fields)
        )
  | EVariant (ty, ctor, args) ->
      VVariant
        (ty, Hashtbl.hash ctor mod 256, List.map (eval_expr state env) args)
  | _ -> assert false

(** Control flow expression evaluation *)
and eval_control_flow state env = function
  | EIf (cond, then_, else_) ->
      if to_bool (eval_expr state env cond) then eval_expr state env then_
      else eval_expr state env else_
  | EMatch (e, cases) ->
      let v = eval_expr state env e in
      let tag =
        match v with
        | VVariant (_, t, _) -> t
        | VInt32 n -> Int32.to_int n
        | _ -> 0
      in
      let rec find_case = function
        | [] ->
            Interp_error.raise_error
              (Pattern_match_failure {context = "EMatch"})
        | (PConstr (name, _), body) :: rest ->
            if Hashtbl.hash name mod 256 = tag then body else find_case rest
        | (PWild, body) :: _ -> body
      in
      eval_expr state env (find_case cases)
  | _ -> assert false

(** Cast and intrinsic expression evaluation *)
and eval_special_expr state env = function
  | EIntrinsic (path, name, args) ->
      let arg_vals = List.map (eval_expr state env) args in
      eval_intrinsic state path name arg_vals
  | ECast (ty, e) -> (
      let v = eval_expr state env e in
      match ty with
      | TInt32 -> VInt32 (to_int32 v)
      | TInt64 -> VInt64 (to_int64 v)
      | TFloat32 -> VFloat32 (to_float32 v)
      | TFloat64 -> VFloat64 (to_float64 v)
      | TBool -> VBool (to_bool v)
      | _ -> v)
  | _ -> assert false

(** Main expression evaluator - dispatches to specialized handlers *)
and eval_expr state env expr =
  match expr with
  (* Simple cases *)
  | EConst (CInt32 n) -> VInt32 n
  | EConst (CInt64 n) -> VInt64 n
  | EConst (CFloat32 f) -> VFloat32 f
  | EConst (CFloat64 f) -> VFloat64 f
  | EConst (CBool b) -> VBool b
  | EConst CUnit -> VUnit
  | EVar v -> lookup_var env v
  | ETuple exprs ->
      VArray (Array.of_list (List.map (eval_expr state env) exprs))
  (* Operators *)
  | EBinop (op, e1, e2) ->
      eval_binop op (eval_expr state env e1) (eval_expr state env e2)
  | EUnop (op, e) -> eval_unop op (eval_expr state env e)
  (* Array operations *)
  | (EArrayRead _ | EArrayReadExpr _ | EArrayLen _ | EArrayCreate _) as e ->
      eval_array_expr state env e
  (* Record/Variant operations *)
  | (ERecordField _ | ERecord _ | EVariant _) as e ->
      eval_composite_expr state env e
  (* Control flow *)
  | (EIf _ | EMatch _) as e -> eval_control_flow state env e
  (* Special operations *)
  | (EIntrinsic _ | ECast _) as e -> eval_special_expr state env e
  (* Function application *)
  | EApp (fn_expr, args) -> eval_app state env fn_expr args

and get_array env name =
  try Hashtbl.find env.arrays name
  with Not_found -> (
    try Hashtbl.find env.shared name
    with Not_found ->
      Interp_error.raise_error (Unbound_variable {name; context = "get_array"}))

and eval_app state env fn_expr args =
  match fn_expr with
  | EIntrinsic (path, name, []) ->
      let arg_vals = List.map (eval_expr state env) args in
      eval_intrinsic state path name arg_vals
  | EVar v -> (
      match Hashtbl.find_opt env.funcs v.var_name with
      | Some hf ->
          (* Call helper function *)
          let arg_vals = List.map (eval_expr state env) args in
          let local_env = copy_env env in
          List.iter2
            (fun param arg -> bind_var local_env param arg)
            hf.hf_params
            arg_vals ;
          (* Execute function body and get return value *)
          exec_stmt_for_return state local_env hf.hf_body
      | None -> Interp_error.raise_error (Unknown_function {name = v.var_name}))
  | _ ->
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "function call";
             reason = "unsupported function expression";
           })

(** {1 Statement Execution} *)

and exec_stmt state env stmt =
  match stmt with
  | SEmpty -> ()
  | SSeq stmts -> List.iter (exec_stmt state env) stmts
  | SAssign (lv, e) ->
      let v = eval_expr state env e in
      assign_lvalue state env lv v
  | SIf (cond, then_s, else_s) ->
      if to_bool (eval_expr state env cond) then exec_stmt state env then_s
      else Option.iter (exec_stmt state env) else_s
  | SWhile (cond, body) ->
      while to_bool (eval_expr state env cond) do
        exec_stmt state env body
      done
  | SFor (v, start, stop, dir, body) ->
      let start_val = to_int32 (eval_expr state env start) in
      let stop_val = to_int32 (eval_expr state env stop) in
      (* OCaml for loops are inclusive: "for i = 0 to n" runs i=0,1,...,n *)
      let incr, cmp =
        match dir with
        | Upto -> ((fun i -> Int32.add i 1l), fun i s -> i <= s)
        | Downto -> ((fun i -> Int32.sub i 1l), fun i s -> i >= s)
      in
      let i = ref start_val in
      while cmp !i stop_val do
        bind_var env v (VInt32 !i) ;
        exec_stmt state env body ;
        i := incr !i
      done
  | SMatch (e, cases) ->
      let v = eval_expr state env e in
      let tag =
        match v with
        | VVariant (_, t, _) -> t
        | VInt32 n -> Int32.to_int n
        | _ -> 0
      in
      let rec find_case = function
        | [] ->
            Interp_error.raise_error
              (Pattern_match_failure {context = "SMatch"})
        | (PConstr (name, vars), body) :: rest ->
            if Hashtbl.hash name mod 256 = tag then begin
              (* Bind pattern variables by name *)
              (match v with
              | VVariant (_, _, args) ->
                  List.iter2
                    (fun vname arg ->
                      Hashtbl.replace env.vars_by_name vname arg)
                    vars
                    args
              | _ -> ()) ;
              body
            end
            else find_case rest
        | (PWild, body) :: _ -> body
      in
      exec_stmt state env (find_case cases)
  | SReturn _ -> () (* Return handled by exec_stmt_for_return *)
  | SBarrier -> Effect.perform Barrier
  | SWarpBarrier -> Effect.perform Barrier
  | SExpr e ->
      let _ = eval_expr state env e in
      ()
  | SLet (v, e, body) -> (
      (* Special handling for shared memory arrays *)
      match e with
      | EArrayCreate (ty, size_expr, Shared) ->
          (* Shared memory: reuse if exists, else create and store in env.shared *)
          let name = v.var_name in
          (match Hashtbl.find_opt env.shared name with
          | Some arr -> bind_var env v (VArray arr)
          | None ->
              let size = to_int (eval_expr state env size_expr) in
              let init =
                match ty with
                | TInt32 -> VInt32 0l
                | TInt64 -> VInt64 0L
                | TFloat32 -> VFloat32 0.0
                | TFloat64 -> VFloat64 0.0
                | TBool -> VBool false
                | _ -> VUnit
              in
              let arr = Array.make size init in
              Hashtbl.add env.shared name arr ;
              bind_var env v (VArray arr)) ;
          exec_stmt state env body
      | _ ->
          let value = eval_expr state env e in
          bind_var env v value ;
          exec_stmt state env body)
  | SLetMut (v, e, body) ->
      let value = eval_expr state env e in
      bind_var env v value ;
      exec_stmt state env body
  | SPragma (_, body) -> exec_stmt state env body
  | SMemFence -> ()
  | SBlock body -> exec_stmt state env body
  | SNative {ocaml; _} ->
      (* Call the typed OCaml fallback *)
      ocaml.run ~block:state.block_dim ~grid:state.grid_dim [||]

and assign_lvalue state env lv value =
  (* Store values directly - VRecord is handled by ERecordField *)
  match lv with
  | LVar v -> bind_var env v value
  | LArrayElem (arr, idx_expr) ->
      let a = get_array env arr in
      let i = to_int (eval_expr state env idx_expr) in
      a.(i) <- value
  | LArrayElemExpr (base_expr, idx_expr) ->
      let a =
        match eval_expr state env base_expr with
        | VArray arr -> arr
        | _ ->
            Interp_error.raise_error
              (Not_an_array {expr = "LArrayElemExpr base"})
      in
      let i = to_int (eval_expr state env idx_expr) in
      a.(i) <- value
  | LRecordField (base_lv, _field) ->
      (* Record field assignment is complex - simplified here *)
      ignore base_lv ;
      Interp_error.raise_error
        (Unsupported_operation
           {
             operation = "record field assignment";
             reason = "not fully supported";
           })

and exec_stmt_for_return state env stmt =
  match stmt with
  | SReturn e -> eval_expr state env e
  | SSeq stmts ->
      let rec exec = function
        | [] -> VUnit
        | [s] -> exec_stmt_for_return state env s
        | s :: rest ->
            exec_stmt state env s ;
            exec rest
      in
      exec stmts
  | SIf (cond, then_s, else_s) -> (
      if to_bool (eval_expr state env cond) then
        exec_stmt_for_return state env then_s
      else
        match else_s with
        | Some s -> exec_stmt_for_return state env s
        | None -> VUnit)
  | SLet (v, e, body) -> (
      (* Special handling for shared memory arrays *)
      match e with
      | EArrayCreate (ty, size_expr, Shared) ->
          let name = v.var_name in
          (match Hashtbl.find_opt env.shared name with
          | Some arr -> bind_var env v (VArray arr)
          | None ->
              let size = to_int (eval_expr state env size_expr) in
              let init =
                match ty with
                | TInt32 -> VInt32 0l
                | TInt64 -> VInt64 0L
                | TFloat32 -> VFloat32 0.0
                | TFloat64 -> VFloat64 0.0
                | TBool -> VBool false
                | _ -> VUnit
              in
              let arr = Array.make size init in
              Hashtbl.add env.shared name arr ;
              bind_var env v (VArray arr)) ;
          exec_stmt_for_return state env body
      | _ ->
          let value = eval_expr state env e in
          bind_var env v value ;
          exec_stmt_for_return state env body)
  | SLetMut (v, e, body) ->
      let value = eval_expr state env e in
      bind_var env v value ;
      exec_stmt_for_return state env body
  | _ ->
      exec_stmt state env stmt ;
      VUnit

(** {1 Kernel Execution} *)

(** Run all threads in a block with BSP barrier synchronization *)
let run_block env body block_idx block_dim grid_dim =
  let bx, by, bz = block_dim in
  let num_threads = bx * by * bz in
  let waiting : (unit, unit) Effect.Deep.continuation option array =
    Array.make num_threads None
  in
  let num_waiting = ref 0 in
  let num_completed = ref 0 in

  let run_thread_with_barrier tid =
    let tx = tid mod bx in
    let ty = tid / bx mod by in
    let tz = tid / (bx * by) in
    let state = {thread_idx = (tx, ty, tz); block_idx; block_dim; grid_dim} in
    let thread_env = copy_env env in
    Effect.Deep.match_with
      (fun () -> exec_stmt state thread_env body)
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

  (* Start all threads *)
  for tid = 0 to num_threads - 1 do
    run_thread_with_barrier tid
  done ;

  (* Superstep loop *)
  while !num_waiting > 0 do
    let to_resume = !num_waiting in
    num_waiting := 0 ;
    for tid = 0 to num_threads - 1 do
      if Option.is_some waiting.(tid) then resume_thread tid
    done ;
    if !num_waiting = to_resume && !num_completed < num_threads then
      Interp_error.raise_error
        (BSP_deadlock {message = "no progress made in interpreter"})
  done

(** Run all blocks in a grid (sequential) *)
let run_grid_sequential env body block_dim grid_dim =
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        Hashtbl.clear env.shared ;
        run_block env body (bx, by, bz) block_dim grid_dim
      done
    done
  done

(** {1 Domain Pool for Parallel Execution} *)

module DomainPool = struct
  type task = unit -> unit

  type t = {
    num_domains : int;
    task_queue : task Queue.t;
    mutex : Mutex.t;
    cond : Condition.t;
    mutable shutdown : bool;
    domains : unit Domain.t array;
    mutable active_tasks : int;
    done_cond : Condition.t;
  }

  let worker pool =
    let rec loop () =
      Mutex.lock pool.mutex ;
      while Queue.is_empty pool.task_queue && not pool.shutdown do
        Condition.wait pool.cond pool.mutex
      done ;
      if pool.shutdown && Queue.is_empty pool.task_queue then begin
        Mutex.unlock pool.mutex
      end
      else begin
        let task = Queue.pop pool.task_queue in
        pool.active_tasks <- pool.active_tasks + 1 ;
        Mutex.unlock pool.mutex ;
        (try task () with _ -> ()) ;
        Mutex.lock pool.mutex ;
        pool.active_tasks <- pool.active_tasks - 1 ;
        if pool.active_tasks = 0 && Queue.is_empty pool.task_queue then
          Condition.broadcast pool.done_cond ;
        Mutex.unlock pool.mutex ;
        loop ()
      end
    in
    loop ()

  let create num_domains =
    let pool =
      {
        num_domains;
        task_queue = Queue.create ();
        mutex = Mutex.create ();
        cond = Condition.create ();
        shutdown = false;
        domains = [||];
        active_tasks = 0;
        done_cond = Condition.create ();
      }
    in
    let domains =
      Array.init num_domains (fun _ -> Domain.spawn (fun () -> worker pool))
    in
    {pool with domains}

  let submit pool task =
    Mutex.lock pool.mutex ;
    Queue.add task pool.task_queue ;
    Condition.signal pool.cond ;
    Mutex.unlock pool.mutex

  let wait_all pool =
    Mutex.lock pool.mutex ;
    while pool.active_tasks > 0 || not (Queue.is_empty pool.task_queue) do
      Condition.wait pool.done_cond pool.mutex
    done ;
    Mutex.unlock pool.mutex
end

(** Global pool - lazily initialized *)
let global_pool : DomainPool.t option ref = ref None

let get_pool () =
  match !global_pool with
  | Some pool -> pool
  | None ->
      let num_cores = try Domain.recommended_domain_count () with _ -> 4 in
      let pool = DomainPool.create num_cores in
      global_pool := Some pool ;
      pool

(** Run all blocks in a grid (parallel - distributes blocks across domain pool)
*)
let run_grid_parallel env body block_dim grid_dim =
  let pool = get_pool () in
  let gx, gy, gz = grid_dim in
  for bz = 0 to gz - 1 do
    for by = 0 to gy - 1 do
      for bx = 0 to gx - 1 do
        (* Shadow loop vars with local bindings to capture values, not refs *)
        let bx = bx and by = by and bz = bz in
        DomainPool.submit pool (fun () ->
            (* Each block gets its own NEW shared memory hashtable.
               Don't use copy_env.shared because it's shared by reference. *)
            let block_env = {(copy_env env) with shared = Hashtbl.create 8} in
            run_block block_env body (bx, by, bz) block_dim grid_dim)
      done
    done
  done ;
  DomainPool.wait_all pool

(** Parallel execution mode flag *)
let parallel_mode = ref true

(** Run all blocks in a grid (uses parallel or sequential based on flag) *)
let run_grid env body block_dim grid_dim =
  if !parallel_mode then run_grid_parallel env body block_dim grid_dim
  else run_grid_sequential env body block_dim grid_dim

(** {1 Public API} *)

(** Argument for kernel execution *)
type arg = ArgArray of value array | ArgScalar of value

(** Run a kernel on CPU *)
let run_kernel (k : kernel) ~block:(bx, by, bz) ~grid:(gx, gy, gz)
    (args : (string * arg) list) =
  let env = create_env () in

  (* Register helper functions *)
  List.iter (fun hf -> Hashtbl.add env.funcs hf.hf_name hf) k.kern_funcs ;

  (* Bind parameters *)
  List.iter2
    (fun decl (name, arg) ->
      match (decl, arg) with
      | DParam (v, Some _), ArgArray arr ->
          Hashtbl.add env.arrays name arr ;
          bind_var env v (VArray arr)
      | DParam (v, None), ArgScalar value -> bind_var env v value
      | DShared (name, ty, Some size_expr), _ ->
          let dummy_state =
            {
              thread_idx = (0, 0, 0);
              block_idx = (0, 0, 0);
              block_dim = (bx, by, bz);
              grid_dim = (gx, gy, gz);
            }
          in
          let size = to_int (eval_expr dummy_state env size_expr) in
          let init =
            match ty with
            | TInt32 -> VInt32 0l
            | TFloat32 -> VFloat32 0.0
            | _ -> VUnit
          in
          Hashtbl.add env.shared name (Array.make size init)
      | _ -> ())
    k.kern_params
    args ;

  run_grid env k.kern_body (bx, by, bz) (gx, gy, gz)

(** {1 V2 Vector Support}

    These functions work with typed Kernel_arg.t values instead of Obj.t. This
    is the preferred interface for Native/Interpreter backends. *)

(** Convert V2 Vector to interpreter value array. Uses the vector's element type
    to create properly typed values. *)
let vector_to_array : type a b. (a, b) Spoc_core.Vector.t -> value array =
 fun vec ->
  let len = Spoc_core.Vector.length vec in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      Array.init len (fun i -> VInt32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      Array.init len (fun i -> VInt64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      Array.init len (fun i -> VFloat32 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      Array.init len (fun i -> VFloat64 (Spoc_core.Vector.get vec i))
  | Spoc_core.Vector.Custom custom -> (
      (* Custom types: use helpers to convert to VRecord *)
      let type_name = custom.Spoc_core.Vector.name in
      match Sarek_type_helpers.lookup type_name with
      | Some h ->
          Array.init len (fun i ->
              let native_record = Spoc_core.Vector.get vec i in
              h.to_value native_record)
      | None ->
          (* Fallback: wrap in VRecord with empty fields - shouldn't happen *)
          Array.init len (fun _i -> VRecord (type_name, [||])))
  | _ ->
      (* Fallback for any other type (e.g., Char) - treat as int32 *)
      Array.init len (fun i -> VInt32 (Obj.magic (Spoc_core.Vector.get vec i)))

(** Write interpreter value array back to V2 Vector *)
let array_to_vector : type a b. value array -> (a, b) Spoc_core.Vector.t -> unit
    =
 fun arr vec ->
  let len = min (Array.length arr) (Spoc_core.Vector.length vec) in
  match Spoc_core.Vector.kind vec with
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Int64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_int64 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float32 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float32 arr.(i))
      done
  | Spoc_core.Vector.Scalar Spoc_core.Vector.Float64 ->
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (to_float64 arr.(i))
      done
  | Spoc_core.Vector.Custom _ ->
      (* Custom types: convert VRecord to native OCaml values using helpers *)
      for i = 0 to len - 1 do
        match arr.(i) with
        | VRecord (type_name, _fields) as vrec -> (
            (* All [@@sarek.type] records have helpers - this should always succeed *)
            match Sarek_type_helpers.lookup type_name with
            | Some h ->
                (* Use generated helper for type-safe conversion *)
                let native_record = h.from_value vrec in
                Spoc_core.Vector.set vec i native_record
            | None ->
                Interp_error.raise_error
                  (Unsupported_operation
                     {
                       operation = "vector_to_array";
                       reason =
                         Printf.sprintf
                           "No helper found for type '%s'. Did you forget \
                            [@@sarek.type]?"
                           type_name;
                     }))
        | _ -> () (* Skip other values *)
      done
  | _ ->
      (* Fallback for any other type (e.g., Char) *)
      for i = 0 to len - 1 do
        Spoc_core.Vector.set vec i (Obj.magic (to_int32 arr.(i)))
      done

(** Existential wrapper to track V2 Vector + its interpreter array for writeback
*)
type writeback =
  | Writeback : (('a, 'b) Spoc_core.Vector.t * value array) -> writeback

(** Convert Kernel_arg.t list to interpreter args, tracking vectors for
    writeback *)
let args_from_kernel_args (k : kernel) (kargs : Spoc_core.Kernel_arg.t list) :
    (string * arg) list * writeback list =
  let writebacks = ref [] in
  let idx = ref 0 in
  let args =
    List.filter_map
      (fun decl ->
        match decl with
        | DParam (v, Some _arr_info) ->
            (* Vector parameter: expects a Vec in Kernel_arg *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              match karg with
              | Spoc_core.Kernel_arg.Vec vec ->
                  let arr = vector_to_array vec in
                  writebacks := Writeback (vec, arr) :: !writebacks ;
                  Some (v.var_name, ArgArray arr)
              | _ ->
                  Interp_error.raise_error
                    (Type_conversion_error
                       {
                         from_type = "scalar";
                         to_type = "Vec";
                         context = "param " ^ v.var_name;
                       })
            end
        | DParam (v, None) ->
            (* Scalar parameter *)
            if !idx >= List.length kargs then None
            else begin
              let karg = List.nth kargs !idx in
              incr idx ;
              let value =
                match karg with
                | Spoc_core.Kernel_arg.Int n -> VInt32 (Int32.of_int n)
                | Spoc_core.Kernel_arg.Int32 n -> VInt32 n
                | Spoc_core.Kernel_arg.Int64 n -> VInt64 n
                | Spoc_core.Kernel_arg.Float32 f -> VFloat32 f
                | Spoc_core.Kernel_arg.Float64 f -> VFloat64 f
                | Spoc_core.Kernel_arg.Vec _ ->
                    Interp_error.raise_error
                      (Type_conversion_error
                         {
                           from_type = "non-scalar";
                           to_type = "scalar";
                           context = "param " ^ v.var_name;
                         })
              in
              Some (v.var_name, ArgScalar value)
            end
        | DShared _ -> None
        | DLocal _ -> None)
      k.kern_params
  in
  (args, List.rev !writebacks)

(** Run kernel with V2 Vector arguments (Kernel_arg.t list). This is the
    preferred entry point for Native/Interpreter backends. Handles conversion
    to/from interpreter format with proper writeback. *)
let run_kernel_with_args (k : kernel) ~(block : int * int * int)
    ~(grid : int * int * int) (kargs : Spoc_core.Kernel_arg.t list) : unit =
  let args, writebacks = args_from_kernel_args k kargs in
  run_kernel k ~block ~grid args ;
  (* Write modified arrays back to V2 Vectors *)
  List.iter (fun (Writeback (vec, arr)) -> array_to_vector arr vec) writebacks
