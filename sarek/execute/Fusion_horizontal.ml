(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(** Horizontal fusion of independent kernels (specs/launch-window.md, FR-011).

    Members [M_0 .. M_{n-1}] with the same block shape become one kernel whose
    grid x is the sum of theirs. Work-group [b] with [off_k <= b < off_k + gx_k]
    runs member [k]'s body unchanged, except that the work-group position it
    observes is rewritten back to its own: [block_idx_x] reads [b - off_k],
    [grid_dim_x] reads [gx_k], and the global index and size intrinsics follow.
    No value flows between members and no arithmetic is added to a member's data
    path, so the result is bit-identical to running the members one after the
    other, on every backend.

    The dispatch condition depends only on [block_idx_x], so it is uniform per
    work-group: barriers inside a member stay legal.

    Every name of member [k] (parameters, locals, shared arrays, loop and
    pattern variables) is prefixed [m<k>_] and its var ids are offset, so
    members cannot capture each other's names. The fused kernel's parameters are
    the members' parameters in order, followed by [hf<k>_off] and [hf<k>_gdx]
    (int32) for each member, a namespace no renamed member name can enter. *)

open Sarek_ir_types

let prefix k name = Printf.sprintf "m%d_%s" k name

(* Var ids of member [k] are shifted into a range of their own. *)
let id_stride = 1 lsl 24

(* The position parameters live OUTSIDE the members' [m<k>_] namespace: a
   member variable named [off] becomes [m<k>_off] and must not shadow them. *)
let int32_var k name =
  {
    var_name = Printf.sprintf "hf%d_%s" k name;
    var_id = (((k + 1) * id_stride) - 1 - if name = "off" then 0 else 1);
    var_type = TInt32;
    var_mutable = false;
  }

let gpu name = EIntrinsic (["Sarek_stdlib"; "Gpu"], name, [])

(** {1 Renaming and position rewrite} *)

type ctx = {k : int; off : expr; gdx : expr}

(* block_idx_x as the member saw it. *)
let member_block ctx = EBinop (Sub, gpu "block_idx_x", ctx.off)

let position ctx name =
  match name with
  | "block_idx_x" -> Some (member_block ctx)
  | "grid_dim_x" -> Some ctx.gdx
  | "global_thread_id" | "global_idx" | "global_idx_x" ->
      (* Same association as the backends' own expansion: tid + bid * bdim. *)
      Some
        (EBinop
           ( Add,
             gpu "thread_idx_x",
             EBinop (Mul, member_block ctx, gpu "block_dim_x") ))
  | "global_size" | "global_size_x" ->
      Some (EBinop (Mul, ctx.gdx, gpu "block_dim_x"))
  | _ -> None

let renaming ctx =
  {
    Ir_rename.name = prefix ctx.k;
    id = (fun i -> i + ((ctx.k + 1) * id_stride));
    position = position ctx;
  }

let rename_var k v =
  Ir_rename.var
    (renaming {k; off = EConst (CInt32 0l); gdx = EConst (CInt32 0l)})
    v

let expr ctx = Ir_rename.expr (renaming ctx)

let stmt ctx = Ir_rename.stmt (renaming ctx)

let decl ctx = Ir_rename.decl (renaming ctx)

(** {1 Assembly} *)

exception Incompatible of string

(* Union of named definitions; the same name must carry the same definition. *)
let union_named ~what name_of members =
  List.fold_left
    (fun acc d ->
      match List.find_opt (fun d' -> name_of d' = name_of d) acc with
      | None -> acc @ [d]
      | Some d' ->
          let same = try d = d' with Invalid_argument _ -> false in
          if same then acc
          else
            raise
              (Incompatible
                 (Printf.sprintf "two different %s named %s" what (name_of d))))
    []
    (List.concat members)

type member_parts = {
  binds : (var * expr) list;  (** hoisted shared arrays, renamed *)
  body : stmt;  (** renamed and rewritten *)
  params : decl list;
  locals : decl list;
  off_var : var;
  gdx_var : var;
}

let member_parts k (m : kernel) =
  let off_var = int32_var k "off" and gdx_var = int32_var k "gdx" in
  let ctx = {k; off = EVar off_var; gdx = EVar gdx_var} in
  let binds, body = Kernel_access.shared_prefix m.kern_body in
  {
    binds = List.map (fun (x, y) -> (rename_var k x, expr ctx y)) binds;
    body = stmt ctx body;
    params = List.map (decl ctx) m.kern_params;
    locals = List.map (decl ctx) m.kern_locals;
    off_var;
    gdx_var;
  }

(* if bx < off_0 + gdx_0 then body_0 else if bx < off_1 + gdx_1 then ... else
   body_{n-1}. The last member needs no test: the grid ends with it. *)
let rec dispatch = function
  | [] -> SEmpty
  | [p] -> p.body
  | p :: rest ->
      let bound = EBinop (Add, EVar p.off_var, EVar p.gdx_var) in
      SIf (EBinop (Lt, gpu "block_idx_x", bound), p.body, Some (dispatch rest))

let scalar_param v = DParam (v, None)

(** Fuse [members] (at least two, each with [Kernel_access.horizontal_verdict] =
    [Ok]) into one kernel named [name]. Raises [Incompatible] when two members
    define the same type or helper name differently. *)
let fuse ~(name : string) (members : kernel list) : kernel =
  let parts = List.mapi member_parts members in
  let body =
    List.fold_right
      (fun (x, y) acc -> SLet (x, y, acc))
      (List.concat_map (fun p -> p.binds) parts)
      (dispatch parts)
  in
  let position_params =
    List.concat_map
      (fun p -> [scalar_param p.off_var; scalar_param p.gdx_var])
      parts
  in
  {
    kern_name = name;
    kern_params = List.concat_map (fun p -> p.params) parts @ position_params;
    kern_locals = List.concat_map (fun p -> p.locals) parts;
    kern_body = body;
    kern_types =
      union_named
        ~what:"record types"
        fst
        (List.map (fun m -> m.kern_types) members);
    kern_variants =
      union_named
        ~what:"variant types"
        fst
        (List.map (fun m -> m.kern_variants) members);
    kern_funcs =
      union_named
        ~what:"helper functions"
        (fun h -> h.hf_name)
        (List.map (fun m -> m.kern_funcs) members);
    kern_native_fn = None;
  }
