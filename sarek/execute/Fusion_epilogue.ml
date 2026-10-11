(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Epilogue fusion: an element-wise consumer inlined after the producer's
    stores (specs/epilogue-fusion.md).

    Producer P stores elements of an intermediate array I; consumer C is
    element-wise ([let i = global_thread_id in body]) and reads I only at its
    own index [i]. In the fused kernel, every store [I[e] <- v] of P becomes

    {[
      let t = e in
      I[t] <- v ;
      if t < tc then (let i = t in body)
    ]}

    where [tc] is the number of threads C was launched with. The thread that
    stored [I[t]] then computes C's element [t], reading [I[t]] back from memory
    just after writing it: the same operations in the same order, with the same
    values. What makes this equal to running P then C is checked elsewhere: P
    stores each element of C's domain exactly once ({!Index_slice}, per launch),
    C reads nothing else P writes, and C writes nothing P or C reads elsewhere
    ({!Launch_window}). The one remaining difference is that the device compiler
    now sees P's store and C's load in one function: a compiler allowed to
    contract [a*b+c] across them could fuse P's last multiply into C's first
    add. That is why this fusion is on by default only where contraction is off
    ([-ffp-contract=off] on HIP).

    Names: P's are prefixed [p_], C's [c_], the new ones [ep_], so no name can
    capture another. C's parameters bound to the same vector as one of P's are
    mapped onto P's parameter (in particular I itself). *)

open Sarek_ir_types

(** {1 Consumer shape} *)

type consumer = {
  index : var;  (** C's [i] *)
  body : stmt;  (** the body under [let i = global id in] *)
  reads_at_index : string list;
      (** array params that C reads only at index [i] (or not at all) *)
  written : string list;  (** array params C writes (always at [i]) *)
}

exception Refuse of string

let refuse fmt = Printf.ksprintf (fun s -> raise (Refuse s)) fmt

let global_ids = ["global_thread_id"; "global_idx"; "global_idx_x"]

(* Every thread/grid position intrinsic the backends accept
   (Sarek_ir_intrinsic_dispatch.thread_intrinsic_names, plus global_size_x). *)
let thread_intrinsics =
  [
    "thread_id_x";
    "thread_idx_x";
    "thread_id_y";
    "thread_idx_y";
    "thread_id_z";
    "thread_idx_z";
    "block_id_x";
    "block_idx_x";
    "block_id_y";
    "block_idx_y";
    "block_id_z";
    "block_idx_z";
    "block_dim_x";
    "block_dim_y";
    "block_dim_z";
    "grid_dim_x";
    "grid_dim_y";
    "grid_dim_z";
    "global_thread_id";
    "global_idx";
    "global_idx_x";
    "global_idx_y";
    "global_idx_z";
    "global_size";
    "global_size_x";
  ]

(* Every array access in [s], as (name, index, is_write). *)
let rec accesses_stmt acc s =
  let ae = accesses_expr and st = accesses_stmt in
  match s with
  | SAssign (LArrayElem (a, i), x) -> ae ((a, Some i, true) :: ae acc i) x
  | SAssign (LVar _, x) -> ae acc x
  | SAssign ((LArrayElemExpr _ | LRecordField _), _) ->
      refuse "store through an expression or a field"
  | SSeq ss -> List.fold_left st acc ss
  | SIf (c, t, f) ->
      Option.fold ~none:(st (ae acc c) t) ~some:(st (st (ae acc c) t)) f
  | SWhile (c, b) -> st (ae acc c) b
  | SFor (_, lo, hi, _, b) -> st (ae (ae acc lo) hi) b
  | SLet (_, x, b) | SLetMut (_, x, b) -> st (ae acc x) b
  | SPragma (_, b) | SBlock b -> st acc b
  | SExpr x -> ae acc x
  | SEmpty -> acc
  | SReturn _ -> refuse "return in the consumer"
  | SBarrier | SWarpBarrier | SMemFence ->
      refuse "barrier or fence in the consumer"
  | SMatch _ -> refuse "match in the consumer"
  | SNative _ | SCoopmat _ -> refuse "native or cooperative-matrix code"

and accesses_expr acc e =
  let ae = accesses_expr in
  match e with
  | EArrayRead (a, i) -> (a, Some i, false) :: ae acc i
  | EArrayReadExpr _ -> refuse "read through an array expression"
  | EArrayCreate _ -> refuse "array creation in the consumer"
  | EIntrinsic (_, name, args) ->
      if Sarek_ir_analysis.is_atomic_intrinsic_name name then refuse "atomic" ;
      if List.mem name thread_intrinsics then
        refuse "consumer reads %s outside its index" name ;
      List.fold_left ae acc args
  | EConst _ | EVar _ | EArrayLen _ -> acc
  | EBinop (_, a, b) -> ae (ae acc a) b
  | EUnop (_, a) | ECast (_, a) | ERecordField (a, _) -> ae acc a
  | ETuple es | EVariant (_, _, es) -> List.fold_left ae acc es
  | EApp (f, es) -> List.fold_left ae acc (f :: es)
  | ERecord (_, fs) -> List.fold_left (fun a (_, x) -> ae a x) acc fs
  | EIf (a, b, c) -> List.fold_left ae acc [a; b; c]
  | EMatch _ -> refuse "match in the consumer"

let is_index (index : var) = function
  | Some (EVar v) -> v.var_id = index.var_id
  | _ -> false

(* Under [Refuse]. *)
let consumer_shape (k : kernel) : consumer =
  (match Kernel_access.horizontal_verdict k with
  | Ok 0 -> ()
  | Ok _ -> refuse "shared memory"
  | Error why -> refuse "%s" why) ;
  if k.kern_locals <> [] then refuse "kernel-scope locals" ;
  (* Helper bodies are not scanned below: a position intrinsic or a barrier
     there would run in the producer's storing thread. *)
  if k.kern_funcs <> [] then refuse "helper functions" ;
  let index, body =
    match k.kern_body with
    | SLet (v, EIntrinsic (["Sarek_stdlib"; "Gpu"], g, []), body)
      when List.mem g global_ids ->
        (v, body)
    | _ -> refuse "not [let i = global_thread_id in ...]"
  in
  let accs = accesses_stmt [] body in
  let params = Kernel_access.vector_params k in
  (* A vector used whole (passed to a helper, aliased) has unknown accesses. *)
  List.iter
    (fun (n, a) ->
      if
        a.Kernel_access.read && a.write
        && not (List.exists (fun (m, _, w) -> m = n && w) accs)
      then refuse "vector %s used as a whole" n)
    (Kernel_access.param_accesses k) ;
  let written =
    List.sort_uniq
      compare
      (List.filter_map (fun (a, _, w) -> if w then Some a else None) accs)
  in
  List.iter
    (fun (a, i, w) ->
      if w && not (is_index index i) then refuse "store to %s off its index" a)
    accs ;
  let reads_at_index =
    List.filter
      (fun p ->
        List.for_all (fun (a, i, w) -> a <> p || w || is_index index i) accs)
      params
  in
  List.iter
    (fun a ->
      if not (List.mem a reads_at_index) then
        refuse "reads %s, which it writes, off its index" a)
    written ;
  {index; body; reads_at_index; written}

let consumer (k : kernel) : (consumer, string) result =
  match consumer_shape k with c -> Ok c | exception Refuse why -> Error why

(** {1 Rewrite} *)

let ep_var id name ty =
  {var_name = "ep_" ^ name; var_id = id; var_type = ty; var_mutable = false}

(* Ids: P keeps its own; C's are shifted above P's largest; the two new
   variables sit above both. The interpreter and [Index_slice] key variables
   by id, so chains (a merged kernel hosting a further epilogue) must never
   reuse an id. *)
type ids = {c_shift : int; tmp_id : int; tc_id : int}

let ids ~producer ~consumer =
  let p = Ir_rename.max_var_id producer and c = Ir_rename.max_var_id consumer in
  let c_shift = p + 1 in
  {c_shift; tmp_id = c_shift + c + 1; tc_id = c_shift + c + 2}

let producer_names =
  {
    Ir_rename.name = (fun n -> "p_" ^ n);
    id = (fun i -> i);
    position = (fun _ -> None);
  }

let consumer_names ~shared ~c_shift =
  {
    Ir_rename.name =
      (fun n ->
        match List.assoc_opt n shared with
        | Some p -> "p_" ^ p
        | None -> "c_" ^ n);
    id = (fun i -> i + c_shift);
    position = (fun _ -> None);
  }

(* Insert the consumer after every store to [array] (already renamed). *)
let rec insert ~array ~tmp_id ~after s =
  let go = insert ~array ~tmp_id ~after in
  match s with
  | SAssign (LArrayElem (a, idx), v) when a = array ->
      let t = ep_var tmp_id "t" TInt32 in
      (* In its own scope: two store sites in one block would otherwise
         declare [ep_t] twice in C. *)
      SBlock
        (SLet
           (t, idx, SSeq [SAssign (LArrayElem (a, EVar t), v); after (EVar t)]))
  | SSeq ss -> SSeq (List.map go ss)
  | SIf (c, t, f) -> SIf (c, go t, Option.map go f)
  | SWhile (c, b) -> SWhile (c, go b)
  | SFor (v, lo, hi, d, b) -> SFor (v, lo, hi, d, go b)
  | SLet (v, x, b) -> SLet (v, x, go b)
  | SLetMut (v, x, b) -> SLetMut (v, x, go b)
  | SPragma (p, b) -> SPragma (p, go b)
  | SBlock b -> SBlock (go b)
  | SMatch (x, cs) -> SMatch (x, List.map (fun (p, b) -> (p, go b)) cs)
  | _ -> s

exception Incompatible of string

(** [fuse ~name ~producer ~p_array ~consumer ~shared] inlines [consumer] after
    [producer]'s stores to [p_array]. [shared] maps every consumer array
    parameter bound to the same vector as a producer parameter (in particular
    the one bound to [p_array]'s vector) to that producer parameter. Parameters:
    P's, then C's that are not shared, then [ep_tc]. *)
let fuse ~name ~(producer : kernel) ~(p_array : string) ~(consumer : kernel)
    ~(shared : (string * string) list) : kernel =
  let c =
    match consumer_shape consumer with
    | c -> c
    | exception Refuse why -> raise (Incompatible why)
  in
  let ids = ids ~producer ~consumer in
  let cn = consumer_names ~shared ~c_shift:ids.c_shift in
  let tc = ep_var ids.tc_id "tc" TInt32 in
  let after t =
    SIf
      ( EBinop (Lt, t, EVar tc),
        SLet (Ir_rename.var cn c.index, t, Ir_rename.stmt cn c.body),
        None )
  in
  let body =
    insert
      ~array:("p_" ^ p_array)
      ~tmp_id:ids.tmp_id
      ~after
      (Ir_rename.stmt producer_names producer.kern_body)
  in
  let c_params =
    List.filter
      (function
        | DParam (v, _) -> not (List.mem_assoc v.var_name shared) | _ -> true)
      consumer.kern_params
  in
  {
    kern_name = name;
    kern_params =
      List.map (Ir_rename.decl producer_names) producer.kern_params
      @ List.map (Ir_rename.decl cn) c_params
      @ [DParam (tc, None)];
    kern_locals = List.map (Ir_rename.decl producer_names) producer.kern_locals;
    kern_body = body;
    kern_types =
      Fusion_horizontal.union_named
        ~what:"record types"
        fst
        [producer.kern_types; consumer.kern_types];
    kern_variants =
      Fusion_horizontal.union_named
        ~what:"variant types"
        fst
        [producer.kern_variants; consumer.kern_variants];
    kern_funcs =
      Fusion_horizontal.union_named
        ~what:"helper functions"
        (fun h -> h.hf_name)
        [producer.kern_funcs; consumer.kern_funcs];
    kern_native_fn = None;
  }
