(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Which outputs a recognised GEMM nest writes, as a function of the launch.
   specs/matrix-pattern-rewrite.md, "Preconditions". *)

open Sarek_ir_types
module Affine = Sarek_ir_affine
module P = Sarek_matrix_pattern
module Eval = Sarek_ir_eval_int

type axis = X | Y | Z

type index_kind =
  | Axis of axis
  | Loop of {var : var; lo : expr; hi : expr}
  | Quotient of {linear : string; divisor : expr}
  | Remainder of {linear : string; divisor : expr}

type role = Row | Col | Batch | Linear

type bound = Upper of expr | Lower of expr

type mapping = {
  m : index_kind;
  n : index_kind;
  batch : index_kind option;
  bounds : (role * bound) list;
}

type region = {me : int; ne : int; ze : int; d : int; t : int}

let ( let* ) = Result.bind

let intrinsic_name = function EIntrinsic (_, n, []) -> Some n | _ -> None

let axis_of_char = function
  | 'x' -> Some X
  | 'y' -> Some Y
  | 'z' -> Some Z
  | _ -> None

let axis_char = function X -> 'x' | Y -> 'y' | Z -> 'z'

let suffix_axis prefix name =
  let lp = String.length prefix in
  if String.length name = lp + 1 && String.sub name 0 lp = prefix then
    axis_of_char name.[lp]
  else None

(* [global_idx_a], or [thread_idx_a + block_dim_a * block_idx_a]. *)
let axis_of_terms terms =
  let named e = Option.value (intrinsic_name e) ~default:"" in
  match terms with
  | [(e, c)] when Affine.poly_is_one c -> suffix_axis "global_idx_" (named e)
  | [(e1, c1); (e2, c2)] -> (
      let pick (e, c) =
        match
          ( suffix_axis "thread_idx_" (named e),
            suffix_axis "block_idx_" (named e) )
        with
        | Some a, _ when Affine.poly_is_one c -> Some (`Thread a)
        | _, Some a -> (
            match c with
            | [{Affine.coef = 1; factors = [f]}]
              when suffix_axis "block_dim_" (named f) = Some a ->
                Some (`Block a)
            | _ -> None)
        | _ -> None
      in
      match (pick (e1, c1), pick (e2, c2)) with
      | Some (`Thread a), Some (`Block b) | Some (`Block b), Some (`Thread a) ->
          if a = b then Some a else None
      | _ -> None)
  | _ -> None

let linear_names = ["global_thread_id"; "global_idx_x"]

let rec loop_bounds v = function
  | SFor (w, lo, hi, Upto, body) ->
      if w.var_id = v.var_id then Some (lo, hi) else loop_bounds v body
  | SSeq l -> List.find_map (loop_bounds v) l
  | SIf (_, t, e) -> (
      match loop_bounds v t with
      | Some b -> Some b
      | None -> Option.bind e (loop_bounds v))
  | SFor (_, _, _, Downto, b)
  | SWhile (_, b)
  | SLet (_, _, b)
  | SLetMut (_, _, b)
  | SPragma (_, b)
  | SBlock b ->
      loop_bounds v b
  | _ -> None

let kind_of_terms (k : kernel) (g : P.gemm) terms =
  match axis_of_terms terms with
  | Some a -> Ok (Axis a)
  | None -> (
      match terms with
      | [(EVar v, c)] when Affine.poly_is_one c -> (
          match
            ( List.exists (fun w -> w.var_id = v.var_id) g.enclosing_loops,
              loop_bounds v k.kern_body )
          with
          | true, Some (lo, hi) -> Ok (Loop {var = v; lo; hi})
          | _ -> Error "an index is a loop variable this rewrite cannot bound")
      | [(EBinop (((Div | Mod) as op), t, divisor), c)]
        when Affine.poly_is_one c -> (
          match intrinsic_name t with
          | Some name when List.mem name linear_names ->
              if op = Div then Ok (Quotient {linear = name; divisor})
              else Ok (Remainder {linear = name; divisor})
          | _ -> Error "an index divides something other than the thread id")
      | _ ->
          Error
            "an index is neither a global index, a loop variable nor the \
             quotient/remainder of the thread id")

let env_of (k : kernel) (g : P.gemm) : Affine.env =
  let scalars =
    List.filter_map
      (function DParam (v, None) -> Some v.var_id | _ -> None)
      k.kern_params
  in
  {
    classify_var =
      (fun v ->
        if List.exists (fun w -> w.var_id = v.var_id) g.enclosing_loops then
          Affine.Atom_leaf
        else if List.mem v.var_id scalars then Affine.Uniform_leaf
        else Affine.Rejected_leaf);
    definition = (fun _ -> None);
  }

let form env e =
  match Affine.of_expr env e with
  | Ok f -> Ok f
  | Error (Affine.Wide_index _) ->
      Error "an index or a guard uses int64 arithmetic"
  | Error _ -> Error "an index or a guard is not affine"

(* ---- Guards ------------------------------------------------------------- *)

let rec conjuncts = function
  | EIf (a, b, EConst (CBool false)) | EBinop (And, a, b) ->
      conjuncts a @ conjuncts b
  | e -> [e]

let flip = function Lt -> Gt | Le -> Ge | Gt -> Lt | Ge -> Le | op -> op

let negate_poly p = Affine.poly_mul (Affine.poly_const (-1)) p

let plus_one e = EBinop (Add, e, EConst (CInt32 1l))

(* [index + c op 0] as a bound on the index. *)
let bound_of_op op c =
  let minus_c = Affine.poly_to_expr (negate_poly c) in
  match op with
  | Lt -> Some (Upper minus_c)
  | Le -> Some (Upper (plus_one minus_c))
  | Gt -> Some (Lower (plus_one minus_c))
  | Ge -> Some (Lower minus_c)
  | _ -> None

(* Atoms compared up to the module path of an intrinsic ([global_thread_id]
   lowered by the PPX carries ["Sarek_stdlib"; "Gpu"]). *)
let rec strip = function
  | EIntrinsic (_, n, args) -> EIntrinsic ([], n, List.map strip args)
  | EBinop (op, a, b) -> EBinop (op, strip a, strip b)
  | EUnop (op, a) -> EUnop (op, strip a)
  | ECast (t, a) -> ECast (t, strip a)
  | e -> e

let same_terms t1 t2 =
  List.map (fun (a, c) -> (strip a, c)) t1
  = List.map (fun (a, c) -> (strip a, c)) t2

let guard_bound env roles cond =
  let try_side op l r =
    let* d = form env (EBinop (Sub, l, r)) in
    Ok
      (List.find_map
         (fun (role, terms) ->
           if same_terms d.Affine.terms terms then
             Option.map (fun b -> (role, b)) (bound_of_op op d.Affine.const)
           else None)
         roles)
  in
  match cond with
  | EBinop (((Lt | Le | Gt | Ge) as op), l, r) -> (
      let* direct = try_side op l r in
      match direct with
      | Some b -> Ok b
      | None -> (
          let* swapped = try_side (flip op) r l in
          match swapped with
          | Some b -> Ok b
          | None -> Error "a guard is not a bound on the row, column or batch"))
  | _ -> Error "a guard is not a comparison"

(* ---- Analysis ------------------------------------------------------------ *)

let terms_of env e =
  let* f = form env e in
  Ok f.Affine.terms

let consistent m n batch =
  match (m, n) with
  | Quotient q, Remainder r ->
      if q.linear = r.linear && q.divisor = r.divisor && batch = None then Ok ()
      else
        Error
          "the quotient and remainder do not share one thread id and divisor"
  | (Quotient _ | Remainder _), _ | _, (Quotient _ | Remainder _) ->
      Error "a quotient index needs the matching remainder as the other index"
  | _ -> (
      let axes =
        List.filter_map
          (function Axis a -> Some a | _ -> None)
          (m :: n :: Option.to_list batch)
      in
      match axes with
      | [a; b] when a = b -> Error "two indices share one launch axis"
      | [a; b; c] when a = b || b = c || a = c ->
          Error "two indices share one launch axis"
      | _ -> Ok ())

(* Every enclosing loop must be the row, column or batch loop: any other loop
   repeats (or skips) the product, which the rewrite would do once. *)
let check_loops (g : P.gemm) kinds =
  let index_vars =
    List.filter_map
      (function Loop {var; _} -> Some var.var_id | _ -> None)
      kinds
  in
  if List.for_all (fun v -> List.mem v.var_id index_vars) g.enclosing_loops then
    Ok ()
  else Error "an enclosing loop is not a row, column or batch index"

let analyse (k : kernel) (g : P.gemm) =
  let* () =
    match g.unmodelled_control with
    | Some what -> Error ("the product sits in " ^ what)
    | None -> Ok ()
  in
  let env = env_of k g in
  let* m_terms = terms_of env g.m_index in
  let* n_terms = terms_of env g.n_index in
  let* m = kind_of_terms k g m_terms in
  let* n = kind_of_terms k g n_terms in
  let* batch, batch_roles =
    match g.batch with
    | [] -> Ok (None, [])
    | [b] -> (
        let* bt = terms_of env b in
        let* kind = kind_of_terms k g bt in
        match kind with
        | Axis _ | Loop _ -> Ok (Some kind, [(Batch, bt)])
        | Quotient _ | Remainder _ ->
            Error "a batch index divides the thread id")
    | _ -> Error "more than one batch index"
  in
  let* () = consistent m n batch in
  let* () = check_loops g (m :: n :: Option.to_list batch) in
  let linear_role =
    match m with
    | Quotient {linear; _} ->
        [
          ( Linear,
            [
              ( EIntrinsic (["Sarek_stdlib"; "Gpu"], linear, []),
                Affine.poly_const 1 );
            ] );
        ]
    | _ -> []
  in
  let roles = [(Row, m_terms); (Col, n_terms)] @ batch_roles @ linear_role in
  let* bounds =
    List.fold_left
      (fun acc c ->
        let* acc = acc in
        let* b = guard_bound env roles c in
        Ok (b :: acc))
      (Ok [])
      (List.concat_map conjuncts g.guards)
  in
  Ok {m; n; batch; bounds = List.rev bounds}

(* ---- The region of one launch ------------------------------------------- *)

let eval launch what e =
  match Eval.eval launch e with
  | Some v -> Ok (Int32.to_int v)
  | None -> Error (Printf.sprintf "%s cannot be evaluated at launch" what)

let threads (launch : Eval.launch) a =
  let pick (x, y, z) = match a with X -> x | Y -> y | Z -> z in
  pick launch.grid * pick launch.block

(* The half-open range [0, hi) an index takes before guards. *)
let range launch ~t = function
  | Axis a -> Ok (threads launch a)
  | Loop {lo; hi; _} ->
      let* lo = eval launch "a loop's lower bound" lo in
      let* hi = eval launch "a loop's upper bound" hi in
      if lo <> 0 then Error "a loop over the output does not start at 0"
      else Ok (hi + 1)
  | Quotient {divisor; _} ->
      let* d = eval launch "the divisor" divisor in
      if d <= 0 then Error "the divisor is not positive"
      else Ok ((t + d - 1) / d)
  | Remainder {divisor; _} ->
      (* Only the columns some thread reaches: [t] may be below [d]. *)
      let* d = eval launch "the divisor" divisor in
      Ok (min d t)

(* Apply the guards on one role to an exclusive upper limit. *)
let bounded launch mapping role limit =
  List.fold_left
    (fun acc (r, b) ->
      let* limit = acc in
      if r <> role then Ok limit
      else
        match b with
        | Upper e ->
            let* u = eval launch "a guard bound" e in
            Ok (min limit u)
        | Lower e ->
            let* l = eval launch "a guard bound" e in
            if l > 0 then Error "a guard excludes the first rows or columns"
            else Ok limit)
    (Ok limit)
    mapping.bounds

let region mapping (launch : Eval.launch) =
  let linear_threads = threads launch X in
  let* t =
    match mapping.m with
    | Quotient _ -> bounded launch mapping Linear linear_threads
    | _ -> Ok 0
  in
  let* m_range = range launch ~t mapping.m in
  let* n_range = range launch ~t mapping.n in
  let* me = bounded launch mapping Row m_range in
  let* ne = bounded launch mapping Col n_range in
  let* ze =
    match mapping.batch with
    | None -> Ok 1
    | Some b ->
        let* z = range launch ~t b in
        bounded launch mapping Batch z
  in
  let* d =
    match mapping.m with
    | Quotient {divisor; _} -> eval launch "the divisor" divisor
    | _ -> Ok 0
  in
  if me <= 0 || ne <= 0 || ze <= 0 then Error "the launch writes no output"
  else Ok {me; ne; ze; d; t}
