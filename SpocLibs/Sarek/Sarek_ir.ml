(** Sarek_ir - Clean intermediate representation for GPU kernels

    This IR separates expressions, statements, and declarations for easier
    analysis and transformation (fusion, AoS-to-SoA, etc.). *)

(** Element types *)
type elttype = TInt32 | TInt64 | TFloat32 | TFloat64 | TBool | TUnit

(** Memory spaces *)
type memspace = Global | Shared | Local

(** Variables with type info *)
type var = {
  var_name : string;
  var_id : int;
  var_type : elttype;
  var_mutable : bool;
}

(** Constants *)
type const =
  | CInt32 of int32
  | CInt64 of int64
  | CFloat32 of float
  | CFloat64 of float
  | CBool of bool
  | CUnit

(** Binary operators *)
type binop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | Eq
  | Ne
  | Lt
  | Le
  | Gt
  | Ge
  | And
  | Or

(** Unary operators *)
type unop = Neg | Not

(** Expressions (pure, no side effects) *)
type expr =
  | EConst of const
  | EVar of var
  | EBinop of binop * expr * expr
  | EUnop of unop * expr
  | EArrayRead of string * expr (* arr[idx] *)
  | ERecordField of expr * string (* r.field *)
  | EIntrinsic of string list * string * expr list (* module path, name, args *)
  | ECast of elttype * expr
  | ETuple of expr list
  | EApp of expr * expr list

(** L-values (assignable locations) *)
type lvalue =
  | LVar of var
  | LArrayElem of string * expr (* arr[idx] *)
  | LRecordField of lvalue * string (* r.field *)

(** Statements (imperative, side effects) *)
type stmt =
  | SAssign of lvalue * expr
  | SSeq of stmt list
  | SIf of expr * stmt * stmt option
  | SWhile of expr * stmt
  | SFor of var * expr * expr * for_dir * stmt
  | SMatch of expr * (pattern * stmt) list
  | SReturn of expr
  | SBarrier (* block_barrier *)
  | SWarpBarrier (* warp-level sync *)
  | SExpr of expr (* side-effecting expression *)
  | SEmpty

and for_dir = Upto | Downto

and pattern =
  | PConstr of string * string list (* Constructor name, bound vars *)
  | PWild

(** Declarations *)
type decl =
  | DParam of
      var * array_info option (* kernel parameter, optional array info *)
  | DLocal of var * expr option (* local variable, optional init *)
  | DShared of
      string * elttype * expr option (* shared array: name, elem type, size *)

and array_info = {arr_elttype : elttype; arr_memspace : memspace}

(** Kernel representation *)
type kernel = {
  kern_name : string;
  kern_params : decl list;
  kern_locals : decl list;
  kern_body : stmt;
}

(** {1 Pretty printing} *)

let string_of_elttype = function
  | TInt32 -> "int32"
  | TInt64 -> "int64"
  | TFloat32 -> "float32"
  | TFloat64 -> "float64"
  | TBool -> "bool"
  | TUnit -> "unit"

let string_of_memspace = function
  | Global -> "global"
  | Shared -> "shared"
  | Local -> "local"

let string_of_binop = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Div -> "/"
  | Mod -> "%"
  | Eq -> "=="
  | Ne -> "!="
  | Lt -> "<"
  | Le -> "<="
  | Gt -> ">"
  | Ge -> ">="
  | And -> "&&"
  | Or -> "||"

let string_of_unop = function Neg -> "-" | Not -> "!"

let pp_var fmt v = Format.fprintf fmt "%s" v.var_name

let rec pp_expr fmt = function
  | EConst (CInt32 n) -> Format.fprintf fmt "%ld" n
  | EConst (CInt64 n) -> Format.fprintf fmt "%LdL" n
  | EConst (CFloat32 f) -> Format.fprintf fmt "%gf" f
  | EConst (CFloat64 f) -> Format.fprintf fmt "%g" f
  | EConst (CBool b) -> Format.fprintf fmt "%b" b
  | EConst CUnit -> Format.fprintf fmt "()"
  | EVar v -> pp_var fmt v
  | EBinop (op, e1, e2) ->
      Format.fprintf fmt "(%a %s %a)" pp_expr e1 (string_of_binop op) pp_expr e2
  | EUnop (op, e) -> Format.fprintf fmt "(%s%a)" (string_of_unop op) pp_expr e
  | EArrayRead (arr, idx) -> Format.fprintf fmt "%s[%a]" arr pp_expr idx
  | ERecordField (e, field) -> Format.fprintf fmt "%a.%s" pp_expr e field
  | EIntrinsic (path, name, args) ->
      let full_name = String.concat "." (path @ [name]) in
      if args = [] then Format.fprintf fmt "%s" full_name
      else Format.fprintf fmt "%s(%a)" full_name pp_exprs args
  | ECast (ty, e) ->
      Format.fprintf fmt "(%s)%a" (string_of_elttype ty) pp_expr e
  | ETuple exprs -> Format.fprintf fmt "(%a)" pp_exprs exprs
  | EApp (fn, args) -> Format.fprintf fmt "%a(%a)" pp_expr fn pp_exprs args

and pp_exprs fmt = function
  | [] -> ()
  | [e] -> pp_expr fmt e
  | e :: es -> Format.fprintf fmt "%a, %a" pp_expr e pp_exprs es

let rec pp_lvalue fmt = function
  | LVar v -> pp_var fmt v
  | LArrayElem (arr, idx) -> Format.fprintf fmt "%s[%a]" arr pp_expr idx
  | LRecordField (lv, field) -> Format.fprintf fmt "%a.%s" pp_lvalue lv field

let rec pp_stmt fmt = function
  | SAssign (lv, e) -> Format.fprintf fmt "%a = %a;" pp_lvalue lv pp_expr e
  | SSeq stmts -> List.iter (fun s -> Format.fprintf fmt "%a@," pp_stmt s) stmts
  | SIf (cond, s1, s2) -> (
      Format.fprintf fmt "@[<v 2>if (%a) {@ %a@]@ }" pp_expr cond pp_stmt s1 ;
      match s2 with
      | None -> ()
      | Some s -> Format.fprintf fmt " else {@ %a@ }" pp_stmt s)
  | SWhile (cond, body) ->
      Format.fprintf
        fmt
        "@[<v 2>while (%a) {@ %a@]@ }"
        pp_expr
        cond
        pp_stmt
        body
  | SFor (v, start, stop, dir, body) ->
      let op = match dir with Upto -> "<" | Downto -> ">" in
      let inc = match dir with Upto -> "++" | Downto -> "--" in
      Format.fprintf
        fmt
        "@[<v 2>for (%s = %a; %s %s %a; %s%s) {@ %a@]@ }"
        v.var_name
        pp_expr
        start
        v.var_name
        op
        pp_expr
        stop
        v.var_name
        inc
        pp_stmt
        body
  | SMatch (e, cases) ->
      Format.fprintf fmt "@[<v 2>match %a with" pp_expr e ;
      List.iter
        (fun (p, s) -> Format.fprintf fmt "@ | %a -> %a" pp_pattern p pp_stmt s)
        cases ;
      Format.fprintf fmt "@]"
  | SReturn e -> Format.fprintf fmt "return %a;" pp_expr e
  | SBarrier -> Format.fprintf fmt "__syncthreads();"
  | SWarpBarrier -> Format.fprintf fmt "__syncwarp();"
  | SExpr e -> Format.fprintf fmt "%a;" pp_expr e
  | SEmpty -> ()

and pp_pattern fmt = function
  | PConstr (name, vars) ->
      if vars = [] then Format.fprintf fmt "%s" name
      else Format.fprintf fmt "%s(%s)" name (String.concat ", " vars)
  | PWild -> Format.fprintf fmt "_"

let pp_decl fmt = function
  | DParam (v, None) ->
      Format.fprintf fmt "%s %s" (string_of_elttype v.var_type) v.var_name
  | DParam (v, Some arr) ->
      Format.fprintf
        fmt
        "%s %s* %s"
        (string_of_memspace arr.arr_memspace)
        (string_of_elttype arr.arr_elttype)
        v.var_name
  | DLocal (v, None) ->
      Format.fprintf fmt "%s %s;" (string_of_elttype v.var_type) v.var_name
  | DLocal (v, Some e) ->
      Format.fprintf
        fmt
        "%s %s = %a;"
        (string_of_elttype v.var_type)
        v.var_name
        pp_expr
        e
  | DShared (name, ty, None) ->
      Format.fprintf fmt "__shared__ %s %s[];" (string_of_elttype ty) name
  | DShared (name, ty, Some size) ->
      Format.fprintf
        fmt
        "__shared__ %s %s[%a];"
        (string_of_elttype ty)
        name
        pp_expr
        size

let pp_kernel fmt k =
  Format.fprintf fmt "@[<v>__kernel void %s(" k.kern_name ;
  (match k.kern_params with
  | [] -> ()
  | [p] -> pp_decl fmt p
  | p :: ps ->
      pp_decl fmt p ;
      List.iter (fun p -> Format.fprintf fmt ", %a" pp_decl p) ps) ;
  Format.fprintf fmt ") {@," ;
  List.iter (fun d -> Format.fprintf fmt "  %a@," pp_decl d) k.kern_locals ;
  Format.fprintf fmt "  %a@," pp_stmt k.kern_body ;
  Format.fprintf fmt "}@]"

(** {1 Conversion from Kirc_Ast.k_ext} *)

exception Conversion_error of string

let conv_error msg = raise (Conversion_error msg)

let elttype_of_kirc : Kirc_Ast.elttype -> elttype = function
  | Kirc_Ast.EInt32 -> TInt32
  | Kirc_Ast.EInt64 -> TInt64
  | Kirc_Ast.EFloat32 -> TFloat32
  | Kirc_Ast.EFloat64 -> TFloat64

let memspace_of_kirc : Kirc_Ast.memspace -> memspace = function
  | Kirc_Ast.Global -> Global
  | Kirc_Ast.Shared -> Shared
  | Kirc_Ast.LocalSpace -> Local

(** Fresh variable ID counter *)
let var_counter = ref 0

let fresh_var_id () =
  incr var_counter ;
  !var_counter

(** Convert expression from k_ext *)
let rec expr_of_k_ext : Kirc_Ast.k_ext -> expr = function
  | Kirc_Ast.Int n -> EConst (CInt32 (Int32.of_int n))
  | Kirc_Ast.Float f -> EConst (CFloat32 f)
  | Kirc_Ast.Double f -> EConst (CFloat64 f)
  | Kirc_Ast.Unit -> EConst CUnit
  | Kirc_Ast.IntVar (id, name, _) ->
      EVar {var_name = name; var_id = id; var_type = TInt32; var_mutable = true}
  | Kirc_Ast.FloatVar (id, name, _) ->
      EVar
        {var_name = name; var_id = id; var_type = TFloat32; var_mutable = true}
  | Kirc_Ast.DoubleVar (id, name, _) ->
      EVar
        {var_name = name; var_id = id; var_type = TFloat64; var_mutable = true}
  | Kirc_Ast.BoolVar (id, name, _) ->
      EVar {var_name = name; var_id = id; var_type = TBool; var_mutable = true}
  | Kirc_Ast.UnitVar (id, name, _) ->
      EVar {var_name = name; var_id = id; var_type = TUnit; var_mutable = true}
  | Kirc_Ast.Id name | Kirc_Ast.IdName name ->
      EVar
        {
          var_name = name;
          var_id = fresh_var_id ();
          var_type = TInt32;
          var_mutable = false;
        }
  (* Binary operators - integer *)
  | Kirc_Ast.Plus (a, b) -> EBinop (Add, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Min (a, b) -> EBinop (Sub, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Mul (a, b) -> EBinop (Mul, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Div (a, b) -> EBinop (Div, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Mod (a, b) -> EBinop (Mod, expr_of_k_ext a, expr_of_k_ext b)
  (* Binary operators - float *)
  | Kirc_Ast.Plusf (a, b) -> EBinop (Add, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Minf (a, b) -> EBinop (Sub, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Mulf (a, b) -> EBinop (Mul, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Divf (a, b) -> EBinop (Div, expr_of_k_ext a, expr_of_k_ext b)
  (* Comparisons *)
  | Kirc_Ast.EqBool (a, b) -> EBinop (Eq, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.LtBool (a, b) -> EBinop (Lt, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.GtBool (a, b) -> EBinop (Gt, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.LtEBool (a, b) -> EBinop (Le, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.GtEBool (a, b) -> EBinop (Ge, expr_of_k_ext a, expr_of_k_ext b)
  (* Logical *)
  | Kirc_Ast.And (a, b) -> EBinop (And, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Or (a, b) -> EBinop (Or, expr_of_k_ext a, expr_of_k_ext b)
  | Kirc_Ast.Not a -> EUnop (Not, expr_of_k_ext a)
  (* Array access *)
  | Kirc_Ast.IntVecAcc (arr, idx) ->
      let arr_name = extract_array_name arr in
      EArrayRead (arr_name, expr_of_k_ext idx)
  | Kirc_Ast.Acc (arr, idx) ->
      let arr_name = extract_array_name arr in
      EArrayRead (arr_name, expr_of_k_ext idx)
  (* Record field *)
  | Kirc_Ast.RecGet (r, field) -> ERecordField (expr_of_k_ext r, field)
  (* Intrinsics *)
  | Kirc_Ast.IntrinsicRef (path, name) -> EIntrinsic (path, name, [])
  | Kirc_Ast.Intrinsics (a, b) -> EIntrinsic ([], a ^ "." ^ b, [])
  (* Application *)
  | Kirc_Ast.App (fn, args) ->
      EApp (expr_of_k_ext fn, Array.to_list (Array.map expr_of_k_ext args))
  (* Global references *)
  | Kirc_Ast.GInt f -> EConst (CInt32 (f ()))
  | Kirc_Ast.GFloat f -> EConst (CFloat32 (f ()))
  | Kirc_Ast.GFloat64 f -> EConst (CFloat64 (f ()))
  | Kirc_Ast.GIntVar name ->
      EVar
        {
          var_name = name;
          var_id = fresh_var_id ();
          var_type = TInt32;
          var_mutable = false;
        }
  | Kirc_Ast.GFloatVar name ->
      EVar
        {
          var_name = name;
          var_id = fresh_var_id ();
          var_type = TFloat32;
          var_mutable = false;
        }
  | Kirc_Ast.GFloat64Var name ->
      EVar
        {
          var_name = name;
          var_id = fresh_var_id ();
          var_type = TFloat64;
          var_mutable = false;
        }
  | Kirc_Ast.CastDoubleVar (id, name) ->
      ECast
        ( TFloat64,
          EVar
            {
              var_name = name;
              var_id = id;
              var_type = TFloat32;
              var_mutable = false;
            } )
  | k ->
      conv_error
        (Printf.sprintf
           "expr_of_k_ext: unsupported %s"
           (Kirc_Ast.string_of_ast k))

and extract_array_name : Kirc_Ast.k_ext -> string = function
  | Kirc_Ast.Id name | Kirc_Ast.IdName name -> name
  | Kirc_Ast.VecVar (_, _, name) -> name
  | Kirc_Ast.Arr (name, _, _, _) -> name
  | k ->
      conv_error
        (Printf.sprintf
           "extract_array_name: not an array %s"
           (Kirc_Ast.string_of_ast k))

(** Convert statement from k_ext *)
let rec stmt_of_k_ext : Kirc_Ast.k_ext -> stmt = function
  | Kirc_Ast.Empty | Kirc_Ast.Unit -> SEmpty
  (* Sequence *)
  | Kirc_Ast.Seq (a, b) -> (
      let s1 = stmt_of_k_ext a in
      let s2 = stmt_of_k_ext b in
      match (s1, s2) with
      | SEmpty, s | s, SEmpty -> s
      | SSeq l1, SSeq l2 -> SSeq (l1 @ l2)
      | SSeq l1, s -> SSeq (l1 @ [s])
      | s, SSeq l2 -> SSeq (s :: l2)
      | _ -> SSeq [s1; s2])
  (* Assignment to array element *)
  | Kirc_Ast.SetV (Kirc_Ast.IntVecAcc (arr, idx), value) ->
      let arr_name = extract_array_name arr in
      SAssign (LArrayElem (arr_name, expr_of_k_ext idx), expr_of_k_ext value)
  | Kirc_Ast.SetV (Kirc_Ast.Acc (arr, idx), value) ->
      let arr_name = extract_array_name arr in
      SAssign (LArrayElem (arr_name, expr_of_k_ext idx), expr_of_k_ext value)
  (* Assignment to variable *)
  | Kirc_Ast.SetV (var, value) ->
      let lv = lvalue_of_k_ext var in
      SAssign (lv, expr_of_k_ext value)
  | Kirc_Ast.Set (var, value) ->
      let lv = lvalue_of_k_ext var in
      SAssign (lv, expr_of_k_ext value)
  (* Record field assignment *)
  | Kirc_Ast.RecSet (r, value) -> (
      match r with
      | Kirc_Ast.RecGet (base, field) ->
          SAssign
            (LRecordField (lvalue_of_k_ext base, field), expr_of_k_ext value)
      | _ -> conv_error "RecSet: expected RecGet")
  (* If-then-else *)
  | Kirc_Ast.Ife (cond, then_, else_) ->
      SIf (expr_of_k_ext cond, stmt_of_k_ext then_, Some (stmt_of_k_ext else_))
  | Kirc_Ast.If (cond, then_) ->
      SIf (expr_of_k_ext cond, stmt_of_k_ext then_, None)
  (* Loop *)
  | Kirc_Ast.DoLoop (init, cond, update, body) ->
      (* Convert to while loop for now; could detect for-loop patterns *)
      let init_stmt = stmt_of_k_ext init in
      let body_stmt = stmt_of_k_ext body in
      let update_stmt = stmt_of_k_ext update in
      SSeq
        [init_stmt; SWhile (expr_of_k_ext cond, SSeq [body_stmt; update_stmt])]
  | Kirc_Ast.While (cond, body) ->
      SWhile (expr_of_k_ext cond, stmt_of_k_ext body)
  (* Match *)
  | Kirc_Ast.Match (_, e, cases) ->
      let cases' = Array.to_list (Array.map convert_case cases) in
      SMatch (expr_of_k_ext e, cases')
  (* Return *)
  | Kirc_Ast.Return e -> SReturn (expr_of_k_ext e)
  (* Local binding *)
  | Kirc_Ast.Local (decl, body) ->
      let _d = decl_of_k_ext decl in
      let body_stmt = stmt_of_k_ext body in
      body_stmt (* TODO: properly handle local declarations *)
  | Kirc_Ast.SetLocalVar (var, init, body) ->
      let v = var_of_k_ext var in
      let init_expr = expr_of_k_ext init in
      let body_stmt = stmt_of_k_ext body in
      SSeq [SAssign (LVar v, init_expr); body_stmt]
  (* Block/Pragma *)
  | Kirc_Ast.Block body -> stmt_of_k_ext body
  | Kirc_Ast.Pragma (_hints, body) ->
      (* TODO: convert pragmas *)
      stmt_of_k_ext body
  (* Declarations treated as statements *)
  | Kirc_Ast.Decl _var -> SEmpty (* declaration only, no init *)
  (* Expression as statement *)
  | k -> (
      (* Try as expression *)
      try SExpr (expr_of_k_ext k)
      with Conversion_error _ ->
        conv_error
          (Printf.sprintf
             "stmt_of_k_ext: unsupported %s"
             (Kirc_Ast.string_of_ast k)))

and lvalue_of_k_ext : Kirc_Ast.k_ext -> lvalue = function
  | Kirc_Ast.IntVar (id, name, _) ->
      LVar {var_name = name; var_id = id; var_type = TInt32; var_mutable = true}
  | Kirc_Ast.FloatVar (id, name, _) ->
      LVar
        {var_name = name; var_id = id; var_type = TFloat32; var_mutable = true}
  | Kirc_Ast.DoubleVar (id, name, _) ->
      LVar
        {var_name = name; var_id = id; var_type = TFloat64; var_mutable = true}
  | Kirc_Ast.BoolVar (id, name, _) ->
      LVar {var_name = name; var_id = id; var_type = TBool; var_mutable = true}
  | Kirc_Ast.UnitVar (id, name, _) ->
      LVar {var_name = name; var_id = id; var_type = TUnit; var_mutable = true}
  | Kirc_Ast.Id name | Kirc_Ast.IdName name ->
      LVar
        {
          var_name = name;
          var_id = fresh_var_id ();
          var_type = TInt32;
          var_mutable = true;
        }
  | Kirc_Ast.IntVecAcc (arr, idx) ->
      LArrayElem (extract_array_name arr, expr_of_k_ext idx)
  | Kirc_Ast.Acc (arr, idx) ->
      LArrayElem (extract_array_name arr, expr_of_k_ext idx)
  | Kirc_Ast.RecGet (base, field) -> LRecordField (lvalue_of_k_ext base, field)
  | k ->
      conv_error
        (Printf.sprintf
           "lvalue_of_k_ext: not an lvalue %s"
           (Kirc_Ast.string_of_ast k))

and var_of_k_ext : Kirc_Ast.k_ext -> var = function
  | Kirc_Ast.IntVar (id, name, _) ->
      {var_name = name; var_id = id; var_type = TInt32; var_mutable = true}
  | Kirc_Ast.FloatVar (id, name, _) ->
      {var_name = name; var_id = id; var_type = TFloat32; var_mutable = true}
  | Kirc_Ast.DoubleVar (id, name, _) ->
      {var_name = name; var_id = id; var_type = TFloat64; var_mutable = true}
  | Kirc_Ast.BoolVar (id, name, _) ->
      {var_name = name; var_id = id; var_type = TBool; var_mutable = true}
  | Kirc_Ast.UnitVar (id, name, _) ->
      {var_name = name; var_id = id; var_type = TUnit; var_mutable = true}
  | Kirc_Ast.Id name | Kirc_Ast.IdName name ->
      {
        var_name = name;
        var_id = fresh_var_id ();
        var_type = TInt32;
        var_mutable = true;
      }
  | k ->
      conv_error
        (Printf.sprintf
           "var_of_k_ext: not a variable %s"
           (Kirc_Ast.string_of_ast k))

and convert_case (tag, binding, body) =
  let pattern =
    match binding with
    | None -> PConstr (string_of_int tag, [])
    | Some (_, name, _, _) -> PConstr (string_of_int tag, [name])
  in
  (pattern, stmt_of_k_ext body)

and decl_of_k_ext : Kirc_Ast.k_ext -> decl = function
  | Kirc_Ast.IntVar (id, name, _) ->
      DLocal
        ( {var_name = name; var_id = id; var_type = TInt32; var_mutable = true},
          None )
  | Kirc_Ast.FloatVar (id, name, _) ->
      DLocal
        ( {var_name = name; var_id = id; var_type = TFloat32; var_mutable = true},
          None )
  | Kirc_Ast.DoubleVar (id, name, _) ->
      DLocal
        ( {var_name = name; var_id = id; var_type = TFloat64; var_mutable = true},
          None )
  | Kirc_Ast.BoolVar (id, name, _) ->
      DLocal
        ( {var_name = name; var_id = id; var_type = TBool; var_mutable = true},
          None )
  | Kirc_Ast.Arr (name, size, elt, memspace) -> (
      match memspace with
      | Kirc_Ast.Shared ->
          DShared (name, elttype_of_kirc elt, Some (expr_of_k_ext size))
      | _ ->
          let v =
            {
              var_name = name;
              var_id = fresh_var_id ();
              var_type = TInt32;
              var_mutable = false;
            }
          in
          DParam
            ( v,
              Some
                {
                  arr_elttype = elttype_of_kirc elt;
                  arr_memspace = memspace_of_kirc memspace;
                } ))
  | k ->
      conv_error
        (Printf.sprintf
           "decl_of_k_ext: not a decl %s"
           (Kirc_Ast.string_of_ast k))

(** Convert full kernel from k_ext *)
let rec of_k_ext : Kirc_Ast.k_ext -> kernel = function
  | Kirc_Ast.Kern (params, body) ->
      let params' = collect_params params in
      let body' = stmt_of_k_ext body in
      {
        kern_name = "kernel";
        kern_params = params';
        kern_locals = [];
        kern_body = body';
      }
  | k ->
      conv_error
        (Printf.sprintf
           "of_k_ext: expected Kern, got %s"
           (Kirc_Ast.string_of_ast k))

and collect_params : Kirc_Ast.k_ext -> decl list = function
  | Kirc_Ast.Params p -> collect_params p
  | Kirc_Ast.Concat (a, b) -> collect_params a @ collect_params b
  | Kirc_Ast.Empty -> []
  | k -> [decl_of_k_ext k]

(** {1 Conversion back to Kirc_Ast.k_ext} *)

let kirc_elttype : elttype -> Kirc_Ast.elttype = function
  | TInt32 -> Kirc_Ast.EInt32
  | TInt64 -> Kirc_Ast.EInt64
  | TFloat32 -> Kirc_Ast.EFloat32
  | TFloat64 -> Kirc_Ast.EFloat64
  | TBool -> Kirc_Ast.EInt32 (* bool as int *)
  | TUnit -> Kirc_Ast.EInt32

let kirc_memspace : memspace -> Kirc_Ast.memspace = function
  | Global -> Kirc_Ast.Global
  | Shared -> Kirc_Ast.Shared
  | Local -> Kirc_Ast.LocalSpace

let rec k_ext_of_expr : expr -> Kirc_Ast.k_ext = function
  | EConst (CInt32 n) -> Kirc_Ast.Int (Int32.to_int n)
  | EConst (CInt64 _) -> conv_error "k_ext_of_expr: int64 not supported"
  | EConst (CFloat32 f) -> Kirc_Ast.Float f
  | EConst (CFloat64 f) -> Kirc_Ast.Double f
  | EConst (CBool true) -> Kirc_Ast.Int 1
  | EConst (CBool false) -> Kirc_Ast.Int 0
  | EConst CUnit -> Kirc_Ast.Unit
  | EVar v -> (
      match v.var_type with
      | TInt32 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable)
      | TInt64 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat32 -> Kirc_Ast.FloatVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat64 -> Kirc_Ast.DoubleVar (v.var_id, v.var_name, v.var_mutable)
      | TBool -> Kirc_Ast.BoolVar (v.var_id, v.var_name, v.var_mutable)
      | TUnit -> Kirc_Ast.UnitVar (v.var_id, v.var_name, v.var_mutable))
  | EBinop (Add, e1, e2) -> Kirc_Ast.Plus (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Sub, e1, e2) -> Kirc_Ast.Min (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Mul, e1, e2) -> Kirc_Ast.Mul (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Div, e1, e2) -> Kirc_Ast.Div (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Mod, e1, e2) -> Kirc_Ast.Mod (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Eq, e1, e2) -> Kirc_Ast.EqBool (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Ne, e1, e2) ->
      Kirc_Ast.Not (Kirc_Ast.EqBool (k_ext_of_expr e1, k_ext_of_expr e2))
  | EBinop (Lt, e1, e2) -> Kirc_Ast.LtBool (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Le, e1, e2) -> Kirc_Ast.LtEBool (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Gt, e1, e2) -> Kirc_Ast.GtBool (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Ge, e1, e2) -> Kirc_Ast.GtEBool (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (And, e1, e2) -> Kirc_Ast.And (k_ext_of_expr e1, k_ext_of_expr e2)
  | EBinop (Or, e1, e2) -> Kirc_Ast.Or (k_ext_of_expr e1, k_ext_of_expr e2)
  | EUnop (Neg, e) -> Kirc_Ast.Min (Kirc_Ast.Int 0, k_ext_of_expr e)
  | EUnop (Not, e) -> Kirc_Ast.Not (k_ext_of_expr e)
  | EArrayRead (arr, idx) ->
      Kirc_Ast.IntVecAcc (Kirc_Ast.Id arr, k_ext_of_expr idx)
  | ERecordField (e, field) -> Kirc_Ast.RecGet (k_ext_of_expr e, field)
  | EIntrinsic (path, name, []) -> Kirc_Ast.IntrinsicRef (path, name)
  | EIntrinsic (path, name, args) ->
      Kirc_Ast.App
        ( Kirc_Ast.IntrinsicRef (path, name),
          Array.of_list (List.map k_ext_of_expr args) )
  | ECast (TFloat64, e) -> (
      match e with
      | EVar v -> Kirc_Ast.CastDoubleVar (v.var_id, v.var_name)
      | _ -> k_ext_of_expr e (* simplified *))
  | ECast (_, e) -> k_ext_of_expr e (* drop other casts *)
  | ETuple _ -> conv_error "k_ext_of_expr: tuple not supported"
  | EApp (fn, args) ->
      Kirc_Ast.App
        (k_ext_of_expr fn, Array.of_list (List.map k_ext_of_expr args))

let rec k_ext_of_stmt : stmt -> Kirc_Ast.k_ext = function
  | SEmpty -> Kirc_Ast.Empty
  | SSeq [] -> Kirc_Ast.Empty
  | SSeq [s] -> k_ext_of_stmt s
  | SSeq (s :: ss) -> Kirc_Ast.Seq (k_ext_of_stmt s, k_ext_of_stmt (SSeq ss))
  | SAssign (LVar v, e) ->
      Kirc_Ast.SetV (k_ext_of_expr (EVar v), k_ext_of_expr e)
  | SAssign (LArrayElem (arr, idx), e) ->
      Kirc_Ast.SetV
        ( Kirc_Ast.IntVecAcc (Kirc_Ast.Id arr, k_ext_of_expr idx),
          k_ext_of_expr e )
  | SAssign (LRecordField (lv, field), e) ->
      Kirc_Ast.RecSet
        (Kirc_Ast.RecGet (k_ext_of_lvalue lv, field), k_ext_of_expr e)
  | SIf (cond, then_, None) ->
      Kirc_Ast.If (k_ext_of_expr cond, k_ext_of_stmt then_)
  | SIf (cond, then_, Some else_) ->
      Kirc_Ast.Ife (k_ext_of_expr cond, k_ext_of_stmt then_, k_ext_of_stmt else_)
  | SWhile (cond, body) ->
      Kirc_Ast.While (k_ext_of_expr cond, k_ext_of_stmt body)
  | SFor (v, start, stop, Upto, body) ->
      (* for v = start to stop do body  =>  v = start; while v < stop do body; v++ *)
      let init = Kirc_Ast.SetV (k_ext_of_expr (EVar v), k_ext_of_expr start) in
      let cond = Kirc_Ast.LtBool (k_ext_of_expr (EVar v), k_ext_of_expr stop) in
      let incr =
        Kirc_Ast.SetV
          ( k_ext_of_expr (EVar v),
            Kirc_Ast.Plus (k_ext_of_expr (EVar v), Kirc_Ast.Int 1) )
      in
      Kirc_Ast.DoLoop (init, cond, incr, k_ext_of_stmt body)
  | SFor (v, start, stop, Downto, body) ->
      let init = Kirc_Ast.SetV (k_ext_of_expr (EVar v), k_ext_of_expr start) in
      let cond = Kirc_Ast.GtBool (k_ext_of_expr (EVar v), k_ext_of_expr stop) in
      let decr =
        Kirc_Ast.SetV
          ( k_ext_of_expr (EVar v),
            Kirc_Ast.Min (k_ext_of_expr (EVar v), Kirc_Ast.Int 1) )
      in
      Kirc_Ast.DoLoop (init, cond, decr, k_ext_of_stmt body)
  | SMatch (e, cases) ->
      let cases' =
        Array.of_list
          (List.mapi
             (fun i (p, s) ->
               let binding =
                 match p with
                 | PConstr (_, []) -> None
                 | PConstr (_, [v]) -> Some ("", v, 0, "")
                 | _ -> None
               in
               (i, binding, k_ext_of_stmt s))
             cases)
      in
      Kirc_Ast.Match ("", k_ext_of_expr e, cases')
  | SReturn e -> Kirc_Ast.Return (k_ext_of_expr e)
  | SBarrier -> Kirc_Ast.IntrinsicRef (["Gpu"], "block_barrier")
  | SWarpBarrier -> Kirc_Ast.IntrinsicRef (["Gpu"], "warp_barrier")
  | SExpr e -> k_ext_of_expr e

and k_ext_of_lvalue : lvalue -> Kirc_Ast.k_ext = function
  | LVar v -> k_ext_of_expr (EVar v)
  | LArrayElem (arr, idx) ->
      Kirc_Ast.IntVecAcc (Kirc_Ast.Id arr, k_ext_of_expr idx)
  | LRecordField (lv, field) -> Kirc_Ast.RecGet (k_ext_of_lvalue lv, field)

let k_ext_of_decl : decl -> Kirc_Ast.k_ext = function
  | DParam (v, None) -> (
      match v.var_type with
      | TInt32 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat32 -> Kirc_Ast.FloatVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat64 -> Kirc_Ast.DoubleVar (v.var_id, v.var_name, v.var_mutable)
      | TBool -> Kirc_Ast.BoolVar (v.var_id, v.var_name, v.var_mutable)
      | TUnit -> Kirc_Ast.UnitVar (v.var_id, v.var_name, v.var_mutable)
      | TInt64 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable))
  | DParam (v, Some _arr) ->
      Kirc_Ast.VecVar (Kirc_Ast.Empty, v.var_id, v.var_name)
  | DLocal (v, _) -> (
      match v.var_type with
      | TInt32 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat32 -> Kirc_Ast.FloatVar (v.var_id, v.var_name, v.var_mutable)
      | TFloat64 -> Kirc_Ast.DoubleVar (v.var_id, v.var_name, v.var_mutable)
      | TBool -> Kirc_Ast.BoolVar (v.var_id, v.var_name, v.var_mutable)
      | TUnit -> Kirc_Ast.UnitVar (v.var_id, v.var_name, v.var_mutable)
      | TInt64 -> Kirc_Ast.IntVar (v.var_id, v.var_name, v.var_mutable))
  | DShared (name, elt, size) ->
      let size_k =
        match size with Some s -> k_ext_of_expr s | None -> Kirc_Ast.Int 0
      in
      Kirc_Ast.Arr (name, size_k, kirc_elttype elt, Kirc_Ast.Shared)

let to_k_ext (k : kernel) : Kirc_Ast.k_ext =
  let params =
    match k.kern_params with
    | [] -> Kirc_Ast.Empty
    | [p] -> k_ext_of_decl p
    | ps ->
        List.fold_left
          (fun acc p -> Kirc_Ast.Concat (acc, k_ext_of_decl p))
          Kirc_Ast.Empty
          ps
  in
  let body = k_ext_of_stmt k.kern_body in
  Kirc_Ast.Kern (Kirc_Ast.Params params, body)
