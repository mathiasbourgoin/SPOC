(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Sarek PPX - intrinsics that type-check but are refused at compile time.
 *
 * Findings C-49 and C-6 of the 2026-10-11 review: no backend lowers the warp
 * votes, the ballot, the active mask or [return_unit] soundly, so the typer
 * refuses them at the use site ({!Sarek_error.Unlowered_intrinsic}). The warp
 * shuffles, [warp_size] and [lane_id] were removed from this table when the
 * backends gained their lowering (docs/plans/warp-shuffle-intake-2026-10-17.md).
 *
 * This module has no dependency on {!Sarek_env}: the environment consults it
 * (to let a user function shadow a refused name) and the typer consults it (to
 * refuse a resolved intrinsic), and neither may depend on the other's use.
 ******************************************************************************)

(** The one table of refused intrinsics, keyed by
    [(module short name, intrinsic name)]: [None] is a core primitive (see
    {!Sarek_core_primitives}), [Some m] an intrinsic registered by a module
    whose SHORT name is [m] (the last component of its module path).

    The typer looks a name up here only AFTER resolving it to an intrinsic, so a
    kernel variable that merely shares a name is never refused.

    The core-primitive entries and their variance/convergence classes stay in
    {!Sarek_core_primitives}: {!Sarek_convergence} keeps checking them, so
    warp-scope supersteps (roadmap S1) start from a checked base.

    COUPLING: a row here has two effects. It refuses the intrinsic, and (through
    {!is_refused_name} and [Sarek_env.add_local_fun]) it makes a kernel-local or
    [[@sarek.module]] function of the same name shadow the intrinsic. Removing a
    row therefore also changes how such a helper resolves: it goes back to
    losing to the intrinsic, as every non-refused intrinsic name does today.
    [test_unlowered.ml] pins this. *)
let table : ((string option * string) * Sarek_error.unlowered_reason) list =
  [
    ((None, "warp_vote_all"), Sarek_error.Warp_collective);
    ((None, "warp_vote_any"), Warp_collective);
    ((None, "warp_ballot"), Warp_collective);
    ((None, "warp_active_mask"), Warp_collective);
    ((Some "Gpu", "return_unit"), Early_return);
  ]

(** The refusal reason for a resolved intrinsic: [module_short] is [None] for a
    core primitive, else the short name of the registering module. *)
let reason ~module_short ~name : Sarek_error.unlowered_reason option =
  List.assoc_opt (module_short, name) table

(** Does any row of {!table} have this intrinsic name? *)
let is_refused_name name = List.exists (fun ((_, n), _) -> n = name) table

(** The refusal for a module-qualified spelling of a refused intrinsic.
    [Gpu.warp_ballot] and [Std.warp_ballot] name the same primitive as the bare
    [warp_ballot], but core primitives are registered under their short name
    only, so the qualified spelling would otherwise be reported as unknown.

    This keys on the module's SHORT name, the last component of the qualifier,
    and only [Gpu] and [Std] qualify ([Std] is an alias of [Gpu], so both reach
    both the core primitives and [Gpu]'s own intrinsics). Another module that
    happens to be called [Gpu] would be treated the same. Returns the
    intrinsic's base name and the reason. *)
let reason_of_qualified_name (qualified : string) :
    (string * Sarek_error.unlowered_reason) option =
  match String.rindex_opt qualified '.' with
  | None -> None
  | Some i ->
      let qualifier = String.sub qualified 0 i in
      let base =
        String.sub qualified (i + 1) (String.length qualified - i - 1)
      in
      let module_short =
        match String.rindex_opt qualifier '.' with
        | None -> qualifier
        | Some j ->
            String.sub qualifier (j + 1) (String.length qualifier - j - 1)
      in
      if module_short <> "Gpu" && module_short <> "Std" then None
      else
        let found =
          match reason ~module_short:None ~name:base with
          | Some r -> Some r
          | None -> reason ~module_short:(Some "Gpu") ~name:base
        in
        Option.map (fun r -> (base, r)) found
