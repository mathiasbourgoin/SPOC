(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for the refusal of unlowered intrinsics (Sarek_unlowered, and its
 * use in Sarek_typer and Sarek_env): warp_* and return_unit type-check but no
 * backend lowers them, so the typer refuses them at the use site.
 ******************************************************************************)

(* Force stdlib initialization to register intrinsics *)
let () = Sarek_stdlib.force_init ()

open Sarek_ast
open Sarek_types
open Sarek_env
open Sarek_typer
open Sarek_typed_ast

let dummy_loc =
  {
    loc_file = "test";
    loc_line = 1;
    loc_col = 0;
    loc_end_line = 1;
    loc_end_col = 0;
    loc_bol = 0;
    loc_end_bol = 0;
  }

let mk_expr e = {e; expr_loc = dummy_loc}

let int32_expr i = mk_expr (EInt32 i)

let bool_expr b = mk_expr (EBool b)

let var_expr s = mk_expr (EVar s)

let unit_expr = mk_expr EUnit

let check_infer_ok msg env expr expected_typ =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  match infer env expr with
  | Ok (te, _) ->
      if repr te.ty <> repr expected_typ then
        Alcotest.failf
          "%s: expected %s, got %s"
          msg
          (typ_to_string expected_typ)
          (typ_to_string te.ty)
  | Error errors ->
      Alcotest.failf
        "%s: type error: %s"
        msg
        (String.concat ", " (List.map Sarek_error.error_to_string errors))

(* Intrinsics that type-check but no backend lowers (C-49, C-6): refused at the
   use site, keyed on the RESOLVED intrinsic, not on spelling. *)
let check_unlowered msg env expr ~name ~reason =
  reset_tvar_counter () ;
  reset_var_id_counter () ;
  match infer env expr with
  | Error [Sarek_error.Unlowered_intrinsic r] ->
      Alcotest.(check string) (msg ^ ": name") name r.name ;
      Alcotest.(check bool) (msg ^ ": reason") true (r.reason = reason)
  | Error errors ->
      Alcotest.failf
        "%s: expected Unlowered_intrinsic, got: %s"
        msg
        (String.concat ", " (List.map Sarek_error.error_to_string errors))
  | Ok (te, _) ->
      Alcotest.failf
        "%s: expected a refusal but it typed as %s"
        msg
        (typ_to_string te.ty)

let warp_primitives =
  List.filter
    (fun (p : Sarek_core_primitives.primitive) -> p.category = "warp")
    Sarek_core_primitives.primitives

let reason_of_warp (p : Sarek_core_primitives.primitive) =
  if p.name = "warp_size" then Sarek_error.Warp_size
  else Sarek_error.Warp_collective

(* Every primitive of category "warp" is refused, so a new warp primitive added
   to the core table cannot silently become usable before a backend lowers it. *)
let test_unlowered_every_warp_primitive () =
  let env = with_stdlib empty in
  Alcotest.(check bool) "warp category is non-empty" true (warp_primitives <> []) ;
  List.iter
    (fun (p : Sarek_core_primitives.primitive) ->
      let reason = reason_of_warp p in
      check_unlowered p.name env (var_expr p.name) ~name:p.name ~reason ;
      check_unlowered
        ("Gpu." ^ p.name)
        env
        (var_expr ("Gpu." ^ p.name))
        ~name:p.name
        ~reason ;
      check_unlowered
        ("Std." ^ p.name)
        env
        (var_expr ("Std." ^ p.name))
        ~name:p.name
        ~reason)
    warp_primitives

let test_unlowered_warp_application () =
  let env = with_stdlib empty in
  let call =
    mk_expr (EApp (var_expr "warp_shuffle", [int32_expr 1l; int32_expr 1l]))
  in
  check_unlowered
    "warp_shuffle applied"
    env
    call
    ~name:"warp_shuffle"
    ~reason:Sarek_error.Warp_collective ;
  let opened = open_module ["Std"] env in
  check_unlowered
    "warp_ballot under open Std"
    opened
    (mk_expr (EApp (var_expr "warp_ballot", [bool_expr true])))
    ~name:"warp_ballot"
    ~reason:Sarek_error.Warp_collective

let test_unlowered_return_unit () =
  let env = with_stdlib empty in
  let reason = Sarek_error.Early_return in
  let call spelling = mk_expr (EApp (var_expr spelling, [unit_expr])) in
  check_unlowered
    "return_unit"
    env
    (call "return_unit")
    ~name:"return_unit"
    ~reason ;
  check_unlowered
    "Gpu.return_unit"
    env
    (call "Gpu.return_unit")
    ~name:"return_unit"
    ~reason ;
  check_unlowered
    "Std.return_unit"
    env
    (call "Std.return_unit")
    ~name:"return_unit"
    ~reason

(* A user binding that merely shares a refused name is NOT refused: the table is
   consulted on the resolved intrinsic, and a variable resolves first. *)
let test_unlowered_not_by_spelling () =
  let env = with_stdlib empty in
  List.iter
    (fun (p : Sarek_core_primitives.primitive) ->
      let local =
        mk_expr (ELet (p.name, None, int32_expr 32l, var_expr p.name))
      in
      check_infer_ok ("local " ^ p.name) env local t_int32)
    warp_primitives ;
  let info =
    {
      vi_type = t_int32;
      vi_mutable = false;
      vi_is_param = true;
      vi_index = 0;
      vi_is_vec = false;
    }
  in
  let with_param = add_var "warp_size" info env in
  check_infer_ok "parameter warp_size" with_param (var_expr "warp_size") t_int32 ;
  let with_param = add_var "return_unit" info env in
  check_infer_ok
    "parameter return_unit"
    with_param
    (var_expr "return_unit")
    t_int32

(* A kernel-local helper named like a refused intrinsic. *)
let test_unlowered_not_by_helper_name () =
  let env = with_stdlib empty in
  let helper name =
    mk_expr
      (ELetRec
         ( name,
           [
             {
               param_name = "q";
               param_type = TEConstr ("int32", []);
               param_loc = dummy_loc;
             };
           ],
           None,
           mk_expr (EBinop (Add, var_expr "q", int32_expr 1l)),
           mk_expr (EApp (var_expr name, [int32_expr 1l])) ))
  in
  check_infer_ok "helper warp_size" env (helper "warp_size") t_int32 ;
  check_infer_ok "helper warp_shuffle" env (helper "warp_shuffle") t_int32 ;
  check_infer_ok "helper return_unit" env (helper "return_unit") t_int32

(* The coupling documented on Sarek_unlowered.table: a row both refuses the
   intrinsic and makes a same-named function shadow it. A name that is NOT in the
   table keeps the existing resolution order, where the intrinsic wins. *)
let test_unlowered_shadowing_coupling () =
  let env = with_stdlib empty in
  let scheme = Sarek_scheme.mono (TFun ([t_int32], t_int32)) in
  let refused = add_local_fun "warp_size" scheme env in
  Alcotest.(check bool)
    "refused name: the intrinsic constant is shadowed"
    true
    (find_intrinsic_const "warp_size" refused = None) ;
  Alcotest.(check bool)
    "refused name: the local function is found"
    true
    (find_local_fun "warp_size" refused <> None) ;
  let refused = add_local_fun "return_unit" scheme env in
  Alcotest.(check bool)
    "refused name: the intrinsic function is shadowed"
    true
    (find_intrinsic_fun "return_unit" refused = None) ;
  let kept = add_local_fun "block_barrier" scheme env in
  Alcotest.(check bool)
    "non-refused name: the intrinsic is still found first (existing order)"
    true
    (find_intrinsic_fun "block_barrier" kept <> None)

let test_unlowered_table_sync () =
  List.iter
    (fun ((module_short, name), reason) ->
      Alcotest.(check bool)
        (name ^ " is found by key")
        true
        (Sarek_unlowered.reason ~module_short ~name = Some reason) ;
      Alcotest.(check bool)
        (name ^ " is a refused name")
        true
        (Sarek_unlowered.is_refused_name name))
    Sarek_unlowered.table ;
  Alcotest.(check bool)
    "an ordinary intrinsic is not refused"
    true
    (Sarek_unlowered.reason ~module_short:None ~name:"block_barrier" = None
    && not (Sarek_unlowered.is_refused_name "block_barrier"))

let () =
  Alcotest.run
    "Sarek_unlowered"
    [
      ( "unlowered intrinsics",
        [
          Alcotest.test_case
            "every warp primitive is refused"
            `Quick
            test_unlowered_every_warp_primitive;
          Alcotest.test_case
            "warp application and open Std"
            `Quick
            test_unlowered_warp_application;
          Alcotest.test_case "return_unit" `Quick test_unlowered_return_unit;
          Alcotest.test_case
            "keyed on the intrinsic, not the spelling"
            `Quick
            test_unlowered_not_by_spelling;
          Alcotest.test_case
            "a helper named like one is not refused"
            `Quick
            test_unlowered_not_by_helper_name;
          Alcotest.test_case
            "shadowing is coupled to the table"
            `Quick
            test_unlowered_shadowing_coupling;
          Alcotest.test_case
            "table lookups agree"
            `Quick
            test_unlowered_table_sync;
        ] );
    ]
