(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Recognition of matrix-product (GEMM) loop nests in Sarek IR. The pattern and
   every rejection reason are specified in specs/matrix-pattern-advice.md. *)

open Sarek_ir_types
module Affine = Sarek_ir_affine
include Sarek_matrix_pattern_types

(* ---- IR helpers -------------------------------------------------------- *)

type item = Bind of var * expr | Item of stmt

let rec flatten = function
  | SSeq l -> List.concat_map flatten l
  | SBlock s | SPragma (_, s) -> flatten s
  | SLet (v, e, body) -> Bind (v, e) :: flatten body
  | SEmpty -> []
  | s -> [Item s]

let same_var v w = v.var_id = w.var_id

let expr_mentions v e =
  Sarek_ir_analysis.expr_fold
    {
      fe =
        (fun found e ->
          found || match e with EVar w -> same_var v w | _ -> false);
      ft = (fun found _ -> found);
      fs = (fun found _ -> found);
      fnative = (fun found -> found);
      visit_lvalue = true;
    }
    false
    e

let stmt_folder ~fe ~fs : 'a Sarek_ir_analysis.folder =
  {fe; ft = (fun x _ -> x); fs; fnative = (fun x -> x); visit_lvalue = true}

let stmt_mentions v s =
  Sarek_ir_analysis.stmt_fold
    (stmt_folder
       ~fe:(fun found e ->
         found || match e with EVar w -> same_var v w | _ -> false)
       ~fs:(fun found s ->
         found
         || match s with SAssign (LVar w, _) -> same_var v w | _ -> false))
    false
    s

let item_mentions v = function
  | Bind (_, e) -> expr_mentions v e
  | Item s -> stmt_mentions v s

let written_arrays s =
  Sarek_ir_analysis.stmt_fold
    (stmt_folder
       ~fe:(fun acc _ -> acc)
       ~fs:(fun acc s ->
         match s with
         | SAssign (LArrayElem (a, _), _) -> a :: acc
         | SCoopmat (CM_store {dst; _}) -> dst :: acc
         | _ -> acc))
    []
    s

let count_mentions v e =
  Sarek_ir_analysis.expr_fold
    (stmt_folder
       ~fe:(fun n e ->
         match e with EVar w when same_var v w -> n + 1 | _ -> n)
       ~fs:(fun n _ -> n))
    0
    e

(* ---- Walk context ------------------------------------------------------ *)

type ctx = {
  arrays : (string * (elttype * memspace)) list;
  uniform_ids : int list;
  lets : (int * expr) list;
  loops : var list;
  guards : expr list;
  written : string list;
}

let bind ctx v e =
  let arrays =
    match e with
    | EArrayCreate (t, _, ms) -> (v.var_name, (t, ms)) :: ctx.arrays
    | _ -> ctx.arrays
  in
  {ctx with lets = (v.var_id, e) :: ctx.lets; arrays}

let bind_items ctx items =
  List.fold_left
    (fun ctx -> function Bind (v, e) -> bind ctx v e | Item _ -> ctx)
    ctx
    items

let affine_env ctx ~atoms : Affine.env =
  {
    classify_var =
      (fun v ->
        if List.exists (same_var v) atoms then Affine.Atom_leaf
        else if List.mem v.var_id ctx.uniform_ids then Affine.Uniform_leaf
        else Affine.Rejected_leaf);
    definition = (fun v -> List.assoc_opt v.var_id ctx.lets);
  }

let global_elt ctx arr =
  match List.assoc_opt arr ctx.arrays with
  | Some (t, Global) -> Ok t
  | _ -> Error (Operand_not_global arr)

let ( let* ) = Result.bind

(* ---- The reduction loop ------------------------------------------------ *)

let rec strip_cast = function ECast (_, e) -> strip_cast e | e -> e

let element_of e =
  match strip_cast e with
  | EArrayRead (arr, idx) -> Ok (arr, idx)
  | other -> Error (Operand_not_an_element other)

let is_acc acc e =
  match strip_cast e with EVar w -> same_var acc w | _ -> false

(* [acc + x * y], [x * y + acc] or [fma x y acc]: the two factors. *)
let factors_of_update acc rhs =
  match rhs with
  | EBinop (Add, l, EBinop (Mul, x, y)) when is_acc acc l -> Ok (x, y, Mul_add)
  | EBinop (Add, EBinop (Mul, x, y), r) when is_acc acc r -> Ok (x, y, Mul_add)
  | EIntrinsic (_, "fma", [x; y; z]) when is_acc acc z ->
      Ok (x, y, Fused_multiply_add)
  | _ -> Error Not_a_product

(* The single [acc := rhs] of a loop body, after its immutable lets. *)
let loop_update acc body =
  let items = flatten body in
  match List.filter (function Item _ -> true | Bind _ -> false) items with
  | [Item (SAssign (LVar w, rhs))] when same_var acc w ->
      let binds =
        List.filter (function Bind _ -> true | Item _ -> false) items
      in
      if List.exists (item_mentions acc) binds then
        Error Loop_body_not_single_update
      else Ok (binds, rhs)
  | _ -> Error Loop_body_not_single_update

let operands ctx acc body =
  let* binds, rhs = loop_update acc body in
  let ctx = bind_items ctx binds in
  let env = affine_env ctx ~atoms:[] in
  let* x, y, update = factors_of_update acc (Affine.inline env rhs) in
  let* ax, ix = element_of x in
  let* ay, iy = element_of y in
  let* ex = global_elt ctx ax in
  let* ey = global_elt ctx ay in
  Ok (ctx, ((ax, ix, ex), (ay, iy, ey)), update)

let uniform_bound ctx e =
  match Affine.of_expr (affine_env ctx ~atoms:ctx.loops) e with
  | Ok f when Affine.is_uniform f ->
      Ok (Affine.inline (affine_env ctx ~atoms:[]) e)
  | Ok _ -> Error Bounds_not_uniform
  | Error (Affine.Data_dependent _) -> Error Bounds_data_dependent
  | Error _ -> Error Bounds_not_uniform

(* ---- The store after the loop ------------------------------------------ *)

type store = {
  st_ctx : ctx;
  st_guard : expr option;
  st_array : string;
  st_index : expr;
  st_value : expr;
}

let store_of ctx acc = function
  | SAssign (LArrayElem (c, ic), e) when expr_mentions acc e ->
      Ok
        {
          st_ctx = ctx;
          st_guard = None;
          st_array = c;
          st_index = ic;
          st_value = e;
        }
  | SIf (g, s, None) when not (expr_mentions acc g) -> (
      (* A guard that reads the accumulator makes the store data-dependent. *)
      let items = flatten s in
      match List.filter (function Item _ -> true | Bind _ -> false) items with
      | [Item (SAssign (LArrayElem (c, ic), e))] when expr_mentions acc e ->
          Ok
            {
              st_ctx = bind_items ctx items;
              st_guard = Some g;
              st_array = c;
              st_index = ic;
              st_value = e;
            }
      | _ -> Error No_output_store)
  | _ -> Error No_output_store

(* Exactly one item after the loop uses [acc], and it is the store. *)
let find_store ctx acc after =
  let rec go ctx = function
    | [] -> Error No_output_store
    | Bind (v, e) :: rest ->
        if expr_mentions acc e then Error Accumulator_escapes
        else go (bind ctx v e) rest
    | Item s :: rest when stmt_mentions acc s ->
        let* st = store_of ctx acc s in
        if List.exists (item_mentions acc) rest then Error Accumulator_escapes
        else Ok st
    | Item _ :: rest -> go ctx rest
  in
  go ctx after

let classify_epilogue env acc st =
  let value = Affine.inline env st.st_value in
  let index = Affine.inline env st.st_index in
  (* Besides [acc], the stored value may only use uniform values (constants,
     scalar parameters, block/grid dimensions, lengths) and the output element
     itself: anything thread-varying would not be an epilogue of the product. *)
  let rec reads_ok = function
    | EArrayRead (arr, idx) -> arr = st.st_array && idx = index
    | EBinop (_, l, r) -> reads_ok l && reads_ok r
    | EUnop (_, e) | ECast (_, e) -> reads_ok e
    | EIntrinsic (_, _, []) as e -> (
        match Affine.of_expr env e with
        | Ok f -> Affine.is_uniform f
        | Error _ -> false)
    | EIntrinsic (_, _, args) -> List.for_all reads_ok args
    | EVar v -> same_var acc v || env.classify_var v = Affine.Uniform_leaf
    | EConst _ | EArrayLen _ -> true
    | _ -> false
  in
  if is_acc acc value then Ok Plain
  else if count_mentions acc value = 1 && reads_ok value then
    Ok (Epilogue value)
  else Error Unsupported_epilogue

(* ---- One candidate loop ---------------------------------------------------- *)

let check_aliasing ctx (ax, _, _) (ay, _, _) c =
  if c = ax || c = ay then Error (Output_aliases_operand c)
  else if List.mem ax ctx.written then Error (Operand_written ax)
  else if List.mem ay ctx.written then Error (Operand_written ay)
  else Ok ()

let analyse_loop ctx acc init ~before ~after = function
  | SFor (_, _, _, Downto, _) -> Error Downto_loop
  | SFor (kv, lo, hi, Upto, body) ->
      if List.exists (item_mentions acc) before then Error Accumulator_escapes
      else
        let* loop_ctx, (ox, oy), update = operands ctx acc body in
        let* k_lo = uniform_bound ctx lo in
        let* k_hi = uniform_bound ctx hi in
        let* st = find_store ctx acc after in
        let env = affine_env st.st_ctx ~atoms:[] in
        let* epilogue = classify_epilogue env acc st in
        let* ec = global_elt st.st_ctx st.st_array in
        let* () = check_aliasing ctx ox oy st.st_array in
        let inl (a, i, e) =
          (a, Affine.inline (affine_env loop_ctx ~atoms:[]) i, e)
        in
        let* a, b, c, (m_index, n_index, batch) =
          Sarek_matrix_roles.index_roles
            (affine_env st.st_ctx ~atoms:(kv :: st.st_ctx.loops))
            kv
            (inl ox, inl oy)
            (st.st_array, Affine.inline env st.st_index, ec)
        in
        let env0 = affine_env ctx ~atoms:[] in
        Ok
          {
            acc;
            init = Affine.inline env0 init;
            k_var = kv;
            k_lo;
            k_hi;
            m_index;
            n_index;
            batch;
            a;
            b;
            c;
            update;
            epilogue;
            guards = List.rev ctx.guards @ Option.to_list st.st_guard;
            enclosing_loops = List.rev ctx.loops;
          }
  | _ -> Error Loop_body_not_single_update

(* ---- Candidates ------------------------------------------------------------ *)

(* Every [for] loop under [s] whose body directly updates [acc] from [acc]. *)
let rec reduction_loops acc s =
  let below = function
    | SSeq l -> List.concat_map (reduction_loops acc) l
    | SIf (_, t, e) ->
        reduction_loops acc t
        @ Option.fold ~none:[] ~some:(reduction_loops acc) e
    | SWhile (_, b)
    | SLet (_, _, b)
    | SLetMut (_, _, b)
    | SPragma (_, b)
    | SBlock b ->
        reduction_loops acc b
    | SFor (_, _, _, _, b) -> reduction_loops acc b
    | SMatch (_, cases) ->
        List.concat_map (fun (_, b) -> reduction_loops acc b) cases
    | _ -> []
  in
  let updates_acc body =
    List.exists
      (function
        | Item (SAssign (LVar w, rhs)) ->
            same_var acc w && expr_mentions acc rhs
        | _ -> false)
      (flatten body)
  in
  match s with
  | SFor (v, _, _, _, body) when updates_acc body -> (v, body, s) :: below s
  | _ -> below s

(* A reduction loop that is not directly in the accumulator's scope. A shared
   or local operand is the more useful reason: it is what a tiled kernel looks
   like. *)
let nested_reason ctx acc body =
  match operands ctx acc body with
  | Error (Operand_not_global _ as r) -> r
  | _ -> Nested_accumulation

let analyse_accumulator ctx acc init body =
  let items = flatten body in
  let top = List.mapi (fun i it -> (i, it)) items in
  List.map
    (fun (kv, loop_body, loop) ->
      let index =
        List.find_map
          (fun (i, it) ->
            match it with Item s when s == loop -> Some i | _ -> None)
          top
      in
      let result =
        match index with
        | None -> Error (nested_reason ctx acc loop_body)
        | Some i ->
            let before = List.filteri (fun j _ -> j < i) items in
            let after = List.filteri (fun j _ -> j > i) items in
            analyse_loop (bind_items ctx before) acc init ~before ~after loop
      in
      match result with
      | Ok g -> Gemm g
      | Error r -> Near_miss {nm_acc = acc; nm_loop = kv; nm_reason = r})
    (reduction_loops acc body)

(* ---- Kernel walk ----------------------------------------------------------- *)

let rec walk ctx = function
  | SLet (v, e, body) -> walk (bind ctx v e) body
  | SLetMut (v, init, body) ->
      analyse_accumulator ctx v init body @ walk ctx body
  | SFor (v, _, _, _, body) -> walk {ctx with loops = v :: ctx.loops} body
  | SIf (g, t, e) ->
      walk {ctx with guards = g :: ctx.guards} t
      @ Option.fold ~none:[] ~some:(walk ctx) e
  | SSeq l -> List.concat_map (walk ctx) l
  | SWhile (_, b) | SPragma (_, b) | SBlock b -> walk ctx b
  | SMatch (_, cases) -> List.concat_map (fun (_, b) -> walk ctx b) cases
  | SAssign _ | SReturn _ | SBarrier | SWarpBarrier | SExpr _ | SEmpty
  | SMemFence | SNative _ | SCoopmat _ ->
      []

let initial_ctx (k : kernel) =
  let param = function
    | DParam (v, Some ai) ->
        `Array (v.var_name, (ai.arr_elttype, ai.arr_memspace))
    | DParam ({var_type = TVec t; var_name; _}, None) ->
        `Array (var_name, (t, Global))
    | DParam (v, None) -> `Scalar v.var_id
    | DShared (name, t, _) -> `Array (name, (t, Shared))
    | DLocal _ -> `Other
  in
  let decls = List.map param (k.kern_params @ k.kern_locals) in
  {
    arrays = List.filter_map (function `Array a -> Some a | _ -> None) decls;
    uniform_ids =
      List.filter_map (function `Scalar id -> Some id | _ -> None) decls;
    lets = [];
    loops = [];
    guards = [];
    written = written_arrays k.kern_body;
  }

let analyze (k : kernel) = walk (initial_ctx k) k.kern_body

let recognise k =
  List.filter_map
    (function Gemm g -> Some g | Near_miss _ -> None)
    (analyze k)
