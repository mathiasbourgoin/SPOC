(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The rewritten kernel of a GEMM nest: SCoopmat tiles staged through shared
   memory. specs/matrix-pattern-rewrite.md, "The rewritten kernel". *)

open Sarek_ir_types
module Affine = Sarek_ir_affine
module P = Sarek_matrix_pattern
module C = Sarek_coopmat_types
open Sarek_matrix_scheme

type lowering = Sarek_matrix_scheme.lowering =
  | Int32_byte_planes
  | F16_operands
  | Bf16_operands
  | Bf16x3_operands

let lowering_name = lowering_name

let extra_names = extra_names

(* ---- The kernel body ----------------------------------------------------- *)

type ctx = {
  fresh : fresh;
  s : scheme;
  g : P.gemm;
  a_addr : address;
  b_addr : address;
  c_addr : address;
  extra : expr list;  (** me, ne, ze, d, t *)
  sa : string list;  (** A staging planes *)
  sb : string list;
  sc : string;  (** accumulator tile *)
  st : string;  (** scratch tile for the integer planes *)
  bk : int;
}

let me c = List.nth c.extra 0

let ne c = List.nth c.extra 1

let d_ c = List.nth c.extra 3

let t_ c = List.nth c.extra 4

let store_planes names idx values =
  SSeq
    (List.map2 (fun name v -> SAssign (LArrayElem (name, idx), v)) names values)

(* Stage one 64 x 16 A tile (or 16 x 64 B tile), zero outside the region. *)
(* [lets [(v, e); ...] body]: a chain of immutable lets. *)
let lets bindings body =
  List.fold_right (fun (v, e) b -> SLet (v, e, b)) bindings body

let local c name = new_var c.fresh name TInt32

(* One element of a staged tile, element [e] of the tile in row-major order:
   the operand element, converted, or zero outside the region. *)
let stage_element c ~kt ~is_a ~e =
  let width = if is_a then c.bk else bn in
  let r = local c "r" and col = local c "c" and o = local c "o" in
  let vr = EVar r and vc = EVar col and vo = EVar o in
  let kk = c.g.k_lo +: (kt *: i32 c.bk) +: if is_a then vc else vr in
  let limit, base =
    if is_a then (me c, gpu "block_idx_y" *: i32 bm)
    else (ne c, gpu "block_idx_x" *: i32 bn)
  in
  let z = gpu "block_idx_z" in
  let arr, idx =
    if is_a then (c.g.a.array, index c.a_addr ~row:vo ~col:kk ~z)
    else (c.g.b.array, index c.b_addr ~row:kk ~col:vo ~z)
  in
  let names = if is_a then c.sa else c.sb in
  (* Padded rows: the shared index of (r, col). *)
  let sidx = (vr *: i32 (width + pad c.s)) +: vc in
  let inside = vo <: limit &&: EBinop (Le, kk, c.g.k_hi) in
  lets
    [
      (r, e /: i32 width);
      (col, e %: i32 width);
      (o, base +: if is_a then vr else vc);
    ]
    (SIf
       ( inside,
         store_planes names sidx (c.s.stage (EArrayRead (arr, idx))),
         Some (store_planes names sidx c.s.zero) ))

let stage_tile c ~tid ~kt ~is_a =
  for_
    c.fresh
    "e8"
    ~upto:((bm * c.bk / threads) - 1)
    (fun e8 ->
      let e = local c "e" in
      lets
        [(e, tid +: (e8 *: i32 threads))]
        (stage_element c ~kt ~is_a ~e:(EVar e)))

let frag c use =
  let a, b, acc, _ = C.fragments_of_config (config c.s) in
  match use with `A -> a | `B -> b | `Acc -> acc

let fa i p = Printf.sprintf "sarek_mm_fa%d_%d" i p

let fb j p = Printf.sprintf "sarek_mm_fb%d_%d" j p

let facc i j s = Printf.sprintf "sarek_mm_acc%d_%d_%d" i j s

let range n = List.init n Fun.id

let ij =
  List.concat_map (fun i -> List.map (fun j -> (i, j)) (range 2)) (range 2)

let decls c =
  List.concat_map
    (fun i ->
      List.map
        (fun p -> SCoopmat (CM_decl {name = fa i p; frag = frag c `A}))
        (range c.s.planes))
    (range 2)
  @ List.concat_map
      (fun j ->
        List.map
          (fun p -> SCoopmat (CM_decl {name = fb j p; frag = frag c `B}))
          (range c.s.planes))
      (range 2)
  @ List.concat_map
      (fun (i, j) ->
        List.map
          (fun s -> SCoopmat (CM_decl {name = facc i j s; frag = frag c `Acc}))
          (range c.s.accumulators))
      ij

let fragment_ops c ~wm ~wn ~ks =
  let load_a i p =
    SCoopmat
      (CM_load
         {
           dst = fa i p;
           frag = frag c `A;
           src = List.nth c.sa p;
           index =
             (((wm *: i32 32) +: i32 (16 * i)) *: i32 (c.bk + pad c.s))
             +: i32 (16 * ks);
           stride = i32 (c.bk + pad c.s);
         })
  and load_b j p =
    SCoopmat
      (CM_load
         {
           dst = fb j p;
           frag = frag c `B;
           src = List.nth c.sb p;
           index =
             i32 (16 * ks * (bn + pad c.s)) +: (wn *: i32 32) +: i32 (16 * j);
           stride = i32 (bn + pad c.s);
         })
  and muladd (i, j) (pa, pb, s) =
    SCoopmat
      (CM_muladd
         {
           dst = facc i j s;
           a = fa i pa;
           b = fb j pb;
           c = facc i j s;
           cfg = config c.s;
         })
  in
  List.concat_map (fun i -> List.map (load_a i) (range c.s.planes)) (range 2)
  @ List.concat_map (fun j -> List.map (load_b j) (range c.s.planes)) (range 2)
  @ List.concat_map (fun q -> List.map (muladd q) c.s.products) ij

let k_loop c ~tid ~wm ~wn =
  let kn = c.g.k_hi -: c.g.k_lo +: i32 1 in
  let steps = (kn +: i32 (c.bk - 1)) /: i32 c.bk in
  let kt = {(new_var c.fresh "kt" TInt32) with var_mutable = true} in
  SFor
    ( kt,
      i32 0,
      steps -: i32 1,
      Upto,
      SSeq
        ([
           SBarrier;
           stage_tile c ~tid ~kt:(EVar kt) ~is_a:true;
           stage_tile c ~tid ~kt:(EVar kt) ~is_a:false;
           SBarrier;
         ]
        @ List.concat_map
            (fun ks -> fragment_ops c ~wm ~wn ~ks)
            (range (c.bk / 16))) )

let store_accs c ~wm ~wn ~s ~dst =
  List.map
    (fun (i, j) ->
      SCoopmat
        (CM_store
           {
             src = facc i j s;
             frag = frag c `Acc;
             dst;
             index =
               (((wm *: i32 32) +: i32 (16 * i)) *: i32 bn)
               +: (wn *: i32 32)
               +: i32 (16 * j);
             stride = i32 bn;
           }))
    ij

(* sc := sum over s of acc_s shifted left by 8s (integer planes). *)
let combine c ~tid ~wm ~wn =
  store_accs c ~wm ~wn ~s:0 ~dst:c.sc
  @ List.concat_map
      (fun s ->
        [SBarrier]
        @ store_accs c ~wm ~wn ~s ~dst:c.st
        @ [
            SBarrier;
            for_
              c.fresh
              "e32"
              ~upto:((bm * bn / threads) - 1)
              (fun e32 ->
                let_
                  c.fresh
                  "e"
                  TInt32
                  (tid +: (e32 *: i32 threads))
                  (fun e ->
                    SAssign
                      ( LArrayElem (c.sc, e),
                        EArrayRead (c.sc, e)
                        +: EBinop (Shl, EArrayRead (c.st, e), i32 (8 * s)) )));
          ])
      (List.init (c.s.accumulators - 1) (fun s -> s + 1))

(* ---- Output ----------------------------------------------------------------- *)

let rec subst ~acc ~c_array ~c_index ~value e =
  let go = subst ~acc ~c_array ~c_index ~value in
  match e with
  | EVar v when v.var_id = acc.var_id -> value
  | EArrayRead (a, _) when a = c_array -> EArrayRead (a, c_index)
  | EBinop (op, x, y) -> EBinop (op, go x, go y)
  | EUnop (op, x) -> EUnop (op, go x)
  | ECast (t, x) -> ECast (t, go x)
  | EIntrinsic (p, n, args) -> EIntrinsic (p, n, List.map go args)
  | e -> e

let is_zero = function
  | EConst (CInt32 0l | CInt64 0L) -> true
  | EConst (CFloat32 f | CFloat64 f) -> f = 0.0 && 1.0 /. f > 0.0
  | _ -> false

(* The value the original stores, with [acc] = init + the tile's sum. *)
let output_value c ~c_index ~sum =
  let g = c.g in
  let s = subst ~acc:g.acc ~c_array:g.c.array ~c_index in
  let init = s ~value:sum g.init in
  let total = if is_zero g.init then sum else init +: sum in
  match g.epilogue with
  | P.Epilogue e -> s ~value:total e
  | P.Plain -> if g.c.elt = g.acc.var_type then total else ECast (g.c.elt, total)

(* Output element [e] of the tile, written when the original writes it. *)
let write_output c ~e =
  let m = local c "m" and n = local c "n" in
  let vm = EVar m and vn = EVar n in
  let c_index = index c.c_addr ~row:vm ~col:vn ~z:(gpu "block_idx_z") in
  let linear_ok =
    EBinop (Or, EBinop (Eq, d_ c, i32 0), (vm *: d_ c) +: vn <: t_ c)
  in
  let covered = vm <: me c &&: (vn <: ne c) &&: linear_ok in
  lets
    [
      (m, (gpu "block_idx_y" *: i32 bm) +: (e /: i32 bn));
      (n, (gpu "block_idx_x" *: i32 bn) +: (e %: i32 bn));
    ]
    (SIf
       ( covered,
         SAssign
           ( LArrayElem (c.g.c.array, c_index),
             output_value c ~c_index ~sum:(EArrayRead (c.sc, e)) ),
         None ))

let write_outputs c ~tid =
  for_
    c.fresh
    "e32"
    ~upto:((bm * bn / threads) - 1)
    (fun e32 ->
      let e = local c "e" in
      lets [(e, tid +: (e32 *: i32 threads))] (write_output c ~e:(EVar e)))

(* ---- Assembly ---------------------------------------------------------------- *)

let shared_array fresh name elt size =
  ( new_var fresh name (TArray (elt, Shared)),
    EArrayCreate (elt, i32 size, Shared) )

let shared_planes fresh prefix elt size n =
  List.init n (fun p ->
      shared_array fresh (Printf.sprintf "%s%d" prefix p) elt size)

let body c0 =
  let fresh = c0.fresh and s = c0.s in
  let tid = local c0 "tid" and wave = local c0 "wave" in
  let wm = local c0 "wm" and wn = local c0 "wn" in
  let sa =
    shared_planes fresh "sa" s.shared_elt (bm * (c0.bk + pad s)) s.planes
  in
  let sb =
    shared_planes fresh "sb" s.shared_elt (c0.bk * (bn + pad s)) s.planes
  in
  let sc = shared_array fresh "sc" s.acc_elt (bm * bn) in
  let st =
    shared_array
      fresh
      "st"
      s.acc_elt
      (if s.accumulators > 1 then bm * bn else 1)
  in
  let name (v, _) = v.var_name in
  let c =
    {
      c0 with
      sa = List.map name sa;
      sb = List.map name sb;
      sc = name sc;
      st = name st;
    }
  in
  let tid_e = EVar tid and wm_e = EVar wm and wn_e = EVar wn in
  lets
    ([
       (tid, gpu "thread_idx_x");
       (wave, EVar tid /: i32 32);
       (wm, EVar wave /: i32 2);
       (wn, EVar wave %: i32 2);
     ]
    @ sa @ sb @ [sc; st])
    (SSeq
       (decls c
       @ [k_loop c ~tid:tid_e ~wm:wm_e ~wn:wn_e; SBarrier]
       @ combine c ~tid:tid_e ~wm:wm_e ~wn:wn_e
       @ [SBarrier; write_outputs c ~tid:tid_e]))

let uses_launch_dims (g : P.gemm) =
  let es =
    [g.k_lo; g.k_hi; g.init]
    @ match g.epilogue with P.Epilogue e -> [e] | P.Plain -> []
  in
  List.exists Sarek_ir_eval_int.mentions_launch_dims es

let address_env (k : kernel) (g : P.gemm) : Affine.env =
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

let ( let* ) = Result.bind

(* The region parameters appended to the original's, under their own names
   (not [new_var]'s prefixed ones: callers pass them by position). *)
let extra_param fresh name =
  fresh.next <- fresh.next + 1 ;
  {var_name = name; var_id = fresh.next; var_type = TInt32; var_mutable = false}

let kernel (k : kernel) (g : P.gemm) lowering =
  let env = address_env k g in
  let batch = match g.batch with [b] -> Some b | _ -> None in
  let* a_addr = address env g.a batch in
  let* b_addr = address env g.b batch in
  let* c_addr = address env g.c batch in
  if uses_launch_dims g then
    Error "a bound, the initial value or the epilogue reads block_dim/grid_dim"
  else
    let sch = scheme ~operand_elt:g.a.elt lowering in
    let fresh = {next = max_var_id k + 1} in
    let extra_vars = List.map (extra_param fresh) extra_names in
    let c =
      {
        fresh;
        s = sch;
        g;
        a_addr;
        b_addr;
        c_addr;
        extra = List.map (fun v -> EVar v) extra_vars;
        sa = [];
        sb = [];
        sc = "";
        st = "";
        bk = choose_bk sch;
      }
    in
    Ok
      {
        k with
        kern_name = k.kern_name ^ "_sarek_mm";
        kern_params =
          k.kern_params @ List.map (fun v -> DParam (v, None)) extra_vars;
        kern_locals = [];
        kern_body = body c;
        kern_funcs = [];
        kern_native_fn = None;
      }

(** Launch geometry of the rewritten kernel for a region. *)
let launch_dims ~me ~ne ~ze =
  ((threads, 1, 1), ((ne + bn - 1) / bn, (me + bm - 1) / bm, ze))

let config_of lowering ~operand_elt = config (scheme ~operand_elt lowering)
