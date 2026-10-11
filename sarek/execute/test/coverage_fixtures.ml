(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Producer/consumer pairs for the epilogue coverage check
 * (Launch_epilogue.covers), shared by its test and its benchmark. Launch
 * records are built directly: [covers] reads only the IR, grid and block.
 ******************************************************************************)

open Sarek_ir_types
module LW = Sarek_execute.Launch_window
module Dims = Spoc_framework.Framework_sig

let gpu name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

let ivar id name =
  {var_name = name; var_id = id; var_type = TInt32; var_mutable = false}

let fvar id name =
  {var_name = name; var_id = id; var_type = TVec TFloat32; var_mutable = false}

let arr = Some {arr_elttype = TFloat32; arr_memspace = Global}

let int n = EConst (CInt32 (Int32.of_int n))

let ( +! ) a b = EBinop (Add, a, b)

let ( *! ) a b = EBinop (Mul, a, b)

let ( /! ) a b = EBinop (Div, a, b)

let ( %! ) a b = EBinop (Mod, a, b)

let ( <! ) a b = EBinop (Lt, a, b)

let ( &&! ) a b = EBinop (And, a, b)

let n_var = ivar 2 "n"

let off_var = ivar 5 "off"

let producer_params =
  [
    DParam (fvar 3 "src", arr);
    DParam (fvar 4 "dst", arr);
    DParam (n_var, None);
    DParam (off_var, None);
  ]

(** [let i = global_thread_id in if i < n then stores i] where [stores i] is a
    statement storing to [dst]. *)
let producer_with ?(name = "p") stores =
  let i = ivar 1 "i" in
  {
    default_kernel with
    kern_name = name;
    kern_params = producer_params;
    kern_body =
      SLet
        ( i,
          gpu "global_thread_id",
          SIf (EVar i <! EVar n_var, stores (EVar i), None) );
  }

let store idx = SAssign (LArrayElem ("dst", idx), EArrayRead ("src", int 0))

(** [dst.(index i off) <- src.(0)] under [i < n]. *)
let producer ?name index =
  producer_with ?name (fun i -> store (index i (EVar off_var)))

(** [dst.(2 i + k)] for k = 0, 1: two stores per thread, through a loop. *)
let loop_producer =
  let k = ivar 6 "k" in
  producer_with ~name:"p_loop" (fun i ->
      SFor (k, int 0, int 1, Upto, store ((int 2 *! i) +! EVar k)))

(** A 2-D tile of an [m x n] row-major output: [row = by*bdy + ty],
    [col = bx*bdx + tx], [dst.(row*n + col)] under [row < m] (and [col < n] when
    [guard_col]). *)
let tile_producer ~guard_col =
  let m = ivar 7 "m" and row = ivar 8 "row" and col = ivar 9 "col" in
  let lane b t d = (gpu b *! gpu d) +! gpu t in
  let cond =
    if guard_col then EVar row <! EVar m &&! (EVar col <! EVar n_var)
    else EVar row <! EVar m
  in
  {
    default_kernel with
    kern_name = (if guard_col then "p_tile" else "p_tile_unguarded");
    kern_params = producer_params @ [DParam (m, None)];
    kern_body =
      SLet
        ( row,
          lane "block_idx_y" "thread_idx_y" "block_dim_y",
          SLet
            ( col,
              lane "block_idx_x" "thread_idx_x" "block_dim_x",
              SIf (cond, store ((EVar row *! EVar n_var) +! EVar col), None) )
        );
  }

(** [let i = global_thread_id in if i < n then y.(i) <- y.(i) +. t.(i)], or
    without the guard. *)
let consumer ?(guarded = true) () =
  let i = ivar 10 "i" and n = ivar 11 "n" in
  let st =
    SAssign
      ( LArrayElem ("y", EVar i),
        EBinop (Add, EArrayRead ("y", EVar i), EArrayRead ("t", EVar i)) )
  in
  {
    default_kernel with
    kern_name = "c";
    kern_params =
      [DParam (fvar 12 "y", arr); DParam (fvar 13 "t", arr); DParam (n, None)];
    kern_body =
      SLet
        ( i,
          gpu "global_thread_id",
          if guarded then SIf (EVar i <! EVar n, st, None) else st );
  }

let launch ?(ir_id = 0) ir ~(grid : Dims.dims) ~(block : Dims.dims) :
    unit LW.launch =
  {
    LW.device_id = 0;
    backend = "probe";
    ir;
    ir_id;
    shared_bytes = None;
    max_grid_x = max_int;
    max_shared = 0;
    args = [];
    block;
    grid;
    shared_mem = 0;
    accesses = [];
    issue = ignore;
    issue_fused = (fun _ _ _ -> ());
  }

let threads n = Dims.dims_1d n

(** 1-D launch of [total] threads in work-groups of [block]. *)
let launch_1d ?ir_id ir ~total ~block =
  launch
    ?ir_id
    ir
    ~grid:(threads ((total + block - 1) / block))
    ~block:(threads block)

let scalars l = List.map (fun (k, v) -> (k, Int32.of_int v)) l

let shape_of c =
  match Sarek_execute.Fusion_epilogue.consumer c.LW.ir with
  | Ok s -> s
  | Error why -> invalid_arg ("consumer refused: " ^ why)

(** The coverage verdict of [c] as the epilogue of [p] on [dst]. *)
let covers ~p ~p_scalars ~c ~c_scalars =
  let shape = shape_of c in
  fst
    (Sarek_execute.Launch_epilogue.covers
       ~p
       ~p_array:"dst"
       ~p_scalars:(scalars p_scalars)
       ~c
       ~shape
       ~c_scalars:(scalars c_scalars))

(** The verdict of each path of the check, separately: the affine proof ([None]
    when the pair is outside its form) and the host enumeration. *)
let both_paths ~p ~p_scalars ~c ~c_scalars =
  let p_scalars = scalars p_scalars and c_scalars = scalars c_scalars in
  let shape = shape_of c in
  let dims (l : _ LW.launch) =
    {
      Sarek_execute.Index_slice.block = (l.block.x, l.block.y, l.block.z);
      grid = (l.grid.x, l.grid.y, l.grid.z);
    }
  in
  let affine =
    match
      Sarek_execute.Coverage_affine.covers
        ~p_body:p.LW.ir.kern_body
        ~p_array:"dst"
        ~p_scalars
        ~p_dims:(dims p)
        ~c_body:c.LW.ir.kern_body
        ~c_written:shape.written
        ~c_scalars
        ~c_dims:(dims c)
    with
    | ok, _ -> Some ok
    | exception Sarek_execute.Coverage_affine.Unknown _ -> None
  in
  let enumerated, _ =
    Sarek_execute.Launch_epilogue.enumerate
      ~p
      ~p_array:"dst"
      ~p_scalars
      ~c
      ~shape
      ~c_scalars
  in
  (affine, enumerated)
