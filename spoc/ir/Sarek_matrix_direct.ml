(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The aligned size class of the matrix rewrite: fragments loaded straight from
   global memory, no staging. specs/matrix-pattern-dynamic.md. *)

open Sarek_ir_types
module P = Sarek_matrix_pattern
module C = Sarek_coopmat_types
open Sarek_matrix_scheme

let ( let* ) = Result.bind

let config =
  {
    C.cfg_shape = {C.m = 16; n = 16; k = 16};
    cfg_a = C.Float16;
    cfg_b = C.Float16;
    cfg_c = C.Float32;
    cfg_result = C.Float32;
    cfg_saturating = false;
    cfg_scope = C.Subgroup;
  }

(* The accumulator must start from zero. Starting from C would load and store
   C in place through fragments: correct on a wave executing in lockstep, but
   the interpreter evaluates a fragment operation once per invocation, so a
   later invocation would read the C an earlier one already stored. Such a
   kernel takes the staged variant, whose epilogue adds C per element. *)
let starts_from_zero (g : P.gemm) =
  match g.init with EConst (CFloat32 f | CFloat64 f) -> f = 0.0 | _ -> false

(** Whether the plan admits the direct variant at all: float16 operands, all
    three matrices row-major, a float32 output stored as is, and an accumulator
    that starts from zero or from the output element. *)
let eligible (g : P.gemm) =
  g.a.elt = TFloat16 && g.b.elt = TFloat16 && g.c.elt = TFloat32
  && g.acc.var_type = TFloat32 && g.a.layout = P.Row_major
  && g.b.layout = P.Row_major && g.c.layout = P.Row_major
  && g.epilogue = P.Plain && starts_from_zero g

(** Whether a region fits the variant: whole 32 x 32 wave tiles, a reduction
    length that is a multiple of 16, a rectangular region. *)
let fits ~me ~ne ~k_extent ~d =
  me mod 32 = 0
  && ne mod 32 = 0
  && k_extent >= 0
  && k_extent mod 16 = 0
  && d = 0

let waves_for ~me ~ne = if me / 32 * (ne / 32) mod 2 = 0 then 2 else 1

let launch_dims ~me ~ne ~ze ~waves =
  ((32 * waves, 1, 1), (me / 32 * (ne / 32) / waves, ze, 1))

let range n = List.init n Fun.id

let ij =
  List.concat_map (fun i -> List.map (fun j -> (i, j)) (range 2)) (range 2)

let fa i = Printf.sprintf "sarek_mm_da%d" i

let fb j = Printf.sprintf "sarek_mm_db%d" j

let facc i j = Printf.sprintf "sarek_mm_dacc%d_%d" i j

let frags = C.fragments_of_config config

let frag_a, frag_b, frag_acc =
  let a, b, acc, _ = frags in
  (a, b, acc)

let cm op = SCoopmat op

(* The wave's tile: rows [row0, row0 + 32), columns [col0, col0 + 32). *)
let tile_body (g : P.gemm) ~a ~b ~c ~row0 ~col0 ~k_steps ~fresh =
  (* Batches on grid y (wave tiles on x). *)
  let z = gpu "block_idx_y" in
  let c_at (i, j) =
    index
      c
      ~row:(EBinop (Add, row0, i32 (16 * i)))
      ~col:(EBinop (Add, col0, i32 (16 * j)))
      ~z
  in
  let decls =
    List.map (fun i -> cm (CM_decl {name = fa i; frag = frag_a})) (range 2)
    @ List.map (fun j -> cm (CM_decl {name = fb j; frag = frag_b})) (range 2)
    @ List.map
        (fun (i, j) -> cm (CM_decl {name = facc i j; frag = frag_acc}))
        ij
  in
  let kk = {(new_var fresh "kk" TInt32) with var_mutable = true} in
  let k_at = EBinop (Add, g.k_lo, EBinop (Mul, EVar kk, i32 16)) in
  let step =
    List.map
      (fun i ->
        cm
          (CM_load
             {
               dst = fa i;
               frag = frag_a;
               src = g.a.array;
               index =
                 index a ~row:(EBinop (Add, row0, i32 (16 * i))) ~col:k_at ~z;
               stride = a.rs;
             }))
      (range 2)
    @ List.map
        (fun j ->
          cm
            (CM_load
               {
                 dst = fb j;
                 frag = frag_b;
                 src = g.b.array;
                 index =
                   index b ~row:k_at ~col:(EBinop (Add, col0, i32 (16 * j))) ~z;
                 stride = b.rs;
               }))
        (range 2)
    @ List.map
        (fun (i, j) ->
          cm
            (CM_muladd
               {dst = facc i j; a = fa i; b = fb j; c = facc i j; cfg = config}))
        ij
  in
  let store (i, j) =
    cm
      (CM_store
         {
           src = facc i j;
           frag = frag_acc;
           dst = g.c.array;
           index = c_at (i, j);
           stride = c.rs;
         })
  in
  SSeq
    (decls
    @ [SFor (kk, i32 0, EBinop (Sub, k_steps, i32 1), Upto, SSeq step)]
    @ List.map store ij)

(** The direct kernel for [waves] waves per block. Parameters: the original's,
    then {!Sarek_matrix_scheme.extra_names} (only [me] and [ne] are read). *)
let kernel (k : kernel) (g : P.gemm) ~waves =
  let env = address_env k g in
  let batch = match g.batch with [b] -> Some b | _ -> None in
  let* a = address env g.a batch in
  let* b = address env g.b batch in
  let* c = address env g.c batch in
  if not (eligible g) then Error "not the direct size class"
  else
    let fresh = {next = max_var_id k + 1} in
    let extra = List.map (extra_param fresh) extra_names in
    let ne = EVar (List.nth extra 1) and me = EVar (List.nth extra 0) in
    let wave = new_var fresh "wave" TInt32
    and row0 = new_var fresh "row0" TInt32 in
    let col0 = new_var fresh "col0" TInt32 in
    let tiles_n = EBinop (Div, ne, i32 32) in
    let k_steps =
      EBinop (Div, EBinop (Add, EBinop (Sub, g.k_hi, g.k_lo), i32 1), i32 16)
    in
    let body =
      SLet
        ( wave,
          EBinop
            ( Add,
              EBinop (Mul, gpu "block_idx_x", i32 waves),
              EBinop (Div, gpu "thread_idx_x", i32 32) ),
          SLet
            ( row0,
              EBinop (Mul, EBinop (Div, EVar wave, tiles_n), i32 32),
              SLet
                ( col0,
                  EBinop (Mul, EBinop (Mod, EVar wave, tiles_n), i32 32),
                  SIf
                    ( EBinop (Lt, EVar row0, me),
                      tile_body
                        g
                        ~a
                        ~b
                        ~c
                        ~row0:(EVar row0)
                        ~col0:(EVar col0)
                        ~k_steps
                        ~fresh,
                      None ) ) ) )
    in
    Ok
      {
        k with
        kern_name = Printf.sprintf "%s_sarek_mm_direct_w%d" k.kern_name waves;
        kern_params = k.kern_params @ List.map (fun v -> DParam (v, None)) extra;
        kern_locals = [];
        kern_body = body;
        kern_funcs = [];
        kern_native_fn = None;
      }
