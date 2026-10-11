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
  subgroup : int;  (** lanes per wave *)
}

let me c = List.nth c.extra 0

let ne c = List.nth c.extra 1

let d_ c = List.nth c.extra 3

let t_ c = List.nth c.extra 4

let threads c = threads_of ~subgroup:c.subgroup

let store_planes names idx values =
  SSeq
    (List.map2 (fun name v -> SAssign (LArrayElem (name, idx), v)) names values)

(* [lets [(v, e); ...] body]: a chain of immutable lets. *)
let lets bindings body =
  List.fold_right (fun (v, e) b -> SLet (v, e, b)) bindings body

let local c name = new_var c.fresh name TInt32

(* ---- Staging ---------------------------------------------------------------

   Both shared tiles are [outer][k] with padded rows: A as [m][k], B
   TRANSPOSED as [n][k], so a fragment of either is 16 contiguous elements
   per lane (B is loaded column-major). Each thread stages [group]
   consecutive k of one row of A (column of B): its [group] shared stores are
   contiguous, and the guarded value is selected, not branched on, so they
   can merge into vector stores. *)

let group = 8

(* Rows of a staged tile: [bm] for A, [bn] for B. *)
let tile_outer ~is_a = if is_a then bm else bn

(* The operand's zero: [stage] of it is the scheme's zero plane values (0 to
   bytes, to f16, to the bf16 split), so an element outside the region stages
   exactly as before. *)
let operand_zero = function
  | TFloat16 -> ECast (TFloat16, fzero)
  | TFloat32 -> fzero
  | elt -> ECast (elt, i32 0)

(* Whether [group] consecutive k are contiguous in the operand's own memory:
   then a thread's [group] reads are contiguous and consecutive threads take
   consecutive groups; otherwise consecutive threads take consecutive rows of
   A (columns of B), so each read instruction is coalesced across threads. *)
let k_contiguous c ~is_a =
  if is_a then c.g.a.layout = P.Row_major else c.g.b.layout = P.Col_major

(* Group [gi] of a staged tile: [group] elements, outer index [outer] (row
   of A, column of B), k from [kg * group], each the operand element
   converted, or zero outside the region. All reads, then all conversions,
   then all stores: a conversion may carry an opaque barrier
   (sarek_f32_barrier) that would otherwise sit between two stores and keep
   them from merging. *)
let stage_group c ~kt ~is_a ~gi =
  let per_row = c.bk / group in
  let outer = local c "outer" and kg = local c "kg" and o = local c "o" in
  let limit, base =
    if is_a then (me c, gpu "block_idx_y" *: i32 bm)
    else (ne c, gpu "block_idx_x" *: i32 bn)
  in
  let z = gpu "block_idx_z" and vo = EVar o in
  let arr, elt =
    if is_a then (c.g.a.array, c.g.a.elt) else (c.g.b.array, c.g.b.elt)
  in
  let names = if is_a then c.sa else c.sb in
  let read j =
    let kk = c.g.k_lo +: (kt *: i32 c.bk) +: (EVar kg *: i32 group) +: i32 j in
    let idx =
      if is_a then index c.a_addr ~row:vo ~col:kk ~z
      else index c.b_addr ~row:kk ~col:vo ~z
    in
    let inside = vo <: limit &&: EBinop (Le, kk, c.g.k_hi) in
    ( new_var c.fresh (Printf.sprintf "v%d" j) elt,
      EIf (inside, EArrayRead (arr, idx), operand_zero elt) )
  in
  let reads = List.init group read in
  (* [planes.(j)]: element j's plane values, one per staging array. *)
  let planes =
    List.mapi
      (fun j (v, _) ->
        List.mapi
          (fun p e ->
            (new_var c.fresh (Printf.sprintf "s%d_%d" j p) c.s.shared_elt, e))
          (c.s.stage (EVar v)))
      reads
  in
  let store j values =
    let sidx =
      (EVar outer *: i32 (c.bk + pad c.s)) +: (EVar kg *: i32 group) +: i32 j
    in
    store_planes names sidx (List.map (fun (s, _) -> EVar s) values)
  in
  let outer_e, kg_e =
    if k_contiguous c ~is_a then (gi /: i32 per_row, gi %: i32 per_row)
    else (gi %: i32 (tile_outer ~is_a), gi /: i32 (tile_outer ~is_a))
  in
  lets
    ([(outer, outer_e); (kg, kg_e); (o, base +: EVar outer)]
    @ reads @ List.concat planes)
    (SSeq (List.mapi store planes))

(* Every thread stages whole groups and every group lies in one row: [bk] is a
   whole number of [group]. The tile has [tile_outer * bk / group] groups; with
   more threads than groups (a 64-wide subgroup: 256 threads, 128 groups) the
   surplus threads are guarded off, with fewer each thread takes several. When
   the threads divide the groups no guard is emitted. *)
let stage_tile c ~tid ~kt ~is_a =
  assert (c.bk mod group = 0) ;
  let threads = threads c in
  let groups = tile_outer ~is_a * c.bk / group in
  for_
    c.fresh
    "g8"
    ~upto:(((groups + threads - 1) / threads) - 1)
    (fun g8 ->
      let gi = local c "gi" in
      let stage = stage_group c ~kt ~is_a ~gi:(EVar gi) in
      let guarded =
        if groups mod threads = 0 then stage
        else SIf (EVar gi <: i32 groups, stage, None)
      in
      lets [(gi, tid +: (g8 *: i32 threads))] guarded)

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
           layout = C.Row_major;
         })
  and load_b j p =
    SCoopmat
      (CM_load
         {
           dst = fb j p;
           frag = frag c `B;
           src = List.nth c.sb p;
           index =
             (((wn *: i32 32) +: i32 (16 * j)) *: i32 (c.bk + pad c.s))
             +: i32 (16 * ks);
           stride = i32 (c.bk + pad c.s);
           layout = C.Col_major;
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
             layout = C.Row_major;
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
              ~upto:((bm * bn / threads c) - 1)
              (fun e32 ->
                let_
                  c.fresh
                  "e"
                  TInt32
                  (tid +: (e32 *: i32 (threads c)))
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
    ~upto:((bm * bn / threads c) - 1)
    (fun e32 ->
      let e = local c "e" in
      lets [(e, tid +: (e32 *: i32 (threads c)))] (write_output c ~e:(EVar e)))

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
    shared_planes fresh "sb" s.shared_elt (bn * (c0.bk + pad s)) s.planes
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
       (wave, EVar tid /: i32 c0.subgroup);
       (wm, EVar wave /: i32 2);
       (wn, EVar wave %: i32 2);
     ]
    @ sa @ sb @ [sc; st])
    (SSeq
       (decls c
       @ [k_loop c ~tid:tid_e ~wm:wm_e ~wn:wn_e; SBarrier]
       @ combine c ~tid:tid_e ~wm:wm_e ~wn:wn_e
       @ [SBarrier; write_outputs c ~tid:tid_e]))

let ( let* ) = Result.bind

let kernel ?(subgroup = 32) (k : kernel) (g : P.gemm) lowering =
  let env = address_env k g in
  let batch = match g.batch with [b] -> Some b | _ -> None in
  let* a_addr = address env g.a batch in
  let* b_addr = address env g.b batch in
  let* c_addr = address env g.c batch in
  if uses_launch_dims g then
    Error "a bound, the initial value or the epilogue reads block_dim/grid_dim"
  else if not (supported_subgroup subgroup) then
    Error
      (Printf.sprintf "no template for %d-wide subgroups (32 and 64)" subgroup)
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
        subgroup;
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
let launch_dims ?(subgroup = 32) ~me ~ne ~ze () =
  ((threads_of ~subgroup, 1, 1), ((ne + bn - 1) / bn, (me + bm - 1) / bm, ze))

let config_of lowering ~operand_elt = config (scheme ~operand_elt lowering)
