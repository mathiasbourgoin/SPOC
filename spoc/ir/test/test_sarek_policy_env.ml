(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The pure half of SAREK_OPT_DISABLE and SAREK_EXPLAIN
   (specs/optimisation-policy.md, M3): the parser, the narrowing of an
   effective policy by restrictions, the explain line format and the events
   about grants that have no effect. Nothing here reads the environment; the
   processes that do are sarek/tests/policy_env.

   The store is process-wide and one-way: every case uses kernel names that no
   other case grants. *)

open Sarek_policy
module E = Env

let restriction_t =
  Alcotest.testable
    (fun ppf -> function
      | E.Disable o -> Format.fprintf ppf "Disable %s" (optimisation_name o)
      | E.Revoke g -> Format.fprintf ppf "Revoke %s" (grant_name g))
    ( = )

let parse_t = Alcotest.(pair (list restriction_t) (list string))

(* ------------------------------------------------------------------------ *)
(* Parser                                                                    *)
(* ------------------------------------------------------------------------ *)

let test_parse_table () =
  List.iter
    (fun (input, expected) ->
      Alcotest.check
        parse_t
        (Printf.sprintf "%S" input)
        expected
        (E.parse input))
    [
      ("", ([], []));
      (" ", ([], []));
      (",", ([], []));
      (" , ,, ", ([], []));
      ("window", ([E.Disable Launch_window], []));
      ("WINDOW", ([E.Disable Launch_window], []));
      ("  Window  ", ([E.Disable Launch_window], []));
      ( "window,epilogue",
        ([E.Disable Launch_window; E.Disable Epilogue_fusion], []) );
      ( " window , , epilogue ,",
        ([E.Disable Launch_window; E.Disable Epilogue_fusion], []) );
      ( "epilogue,window",
        ([E.Disable Epilogue_fusion; E.Disable Launch_window], []) );
      ("window,window", ([E.Disable Launch_window], []));
      ("window,WINDOW, window", ([E.Disable Launch_window], []));
      ("bogus", ([], ["bogus"]));
      ("Bogus", ([], ["bogus"]));
      ( "window,bogus,unroll",
        ([E.Disable Launch_window; E.Disable Auto_unroll], ["bogus"]) );
      ("bogus,other,bogus", ([], ["bogus"; "other"]));
      ("all", ([], ["all"]));
      ("window;unroll", ([], ["window;unroll"]));
      ("window unroll", ([], ["window unroll"]));
      ("single-rounding", ([E.Revoke (Deviation Single_rounding)], []));
      ( "Single-Rounding, matrix",
        ([E.Revoke (Deviation Single_rounding); E.Disable Matrix_rewrite], [])
      );
    ]

let test_parse_every_name () =
  List.iter
    (fun o ->
      Alcotest.check
        parse_t
        (optimisation_name o)
        ([E.Disable o], [])
        (E.parse (optimisation_name o)))
    all_optimisations ;
  List.iter
    (fun g ->
      Alcotest.check
        parse_t
        (grant_name g)
        ([E.Revoke g], [])
        (E.parse (grant_name g)))
    all_grants ;
  Alcotest.check
    parse_t
    "all twelve at once, in vocabulary order"
    ( List.map (fun o -> E.Disable o) all_optimisations
      @ List.map (fun g -> E.Revoke g) all_grants,
      [] )
    (E.parse
       (String.concat
          ","
          (List.map optimisation_name all_optimisations
          @ List.map grant_name all_grants)))

let test_names_distinct () =
  let names =
    List.map optimisation_name all_optimisations
    @ List.map grant_name all_grants
  in
  Alcotest.(check int)
    "no name is both an optimisation and a grant"
    (List.length names)
    (List.length (List.sort_uniq compare names))

let test_unknown_message () =
  let m = E.unknown_message "bogus" in
  let has needle = Astring.String.is_infix ~affix:needle m in
  Alcotest.(check bool) "names the variable" true (has "SAREK_OPT_DISABLE") ;
  Alcotest.(check bool) "names the item" true (has "\"bogus\"") ;
  List.iter
    (fun n -> Alcotest.(check bool) ("lists " ^ n) true (has n))
    (List.map optimisation_name all_optimisations
    @ List.map grant_name all_grants) ;
  Alcotest.(check bool) "one line" false (String.contains m '\n')

(* ------------------------------------------------------------------------ *)
(* Restrictions only restrict                                                *)
(* ------------------------------------------------------------------------ *)

let items =
  List.map (fun o -> E.Disable o) all_optimisations
  @ List.map (fun g -> E.Revoke g) all_grants

let subsets =
  List.fold_left (fun acc x -> acc @ List.map (fun s -> x :: s) acc) [[]] items

let full_budget =
  List.fold_left
    (fun b g ->
      match Budget.add g ~reason:("r:" ^ grant_name g) b with
      | Ok b -> b
      | Error _ -> Alcotest.fail "add refused")
    Budget.exact
    all_grants

let backends =
  ["HIP"; "Metal"; "CUDA/PTX"; "CUDA/C"; "OpenCL"; "Vulkan"; "Native"]

let applied = function Applied _ -> true | Refused _ -> false

(* No value of SAREK_OPT_DISABLE grants anything: against a budget that holds
   every grant and against the empty one, narrowing never adds a grant, never
   removes a restriction, and never turns a refused decision into an applied
   one. 2^12 subsets. *)
let test_narrow_never_grants () =
  List.iter
    (fun base_budget ->
      let base = {budget = base_budget; disabled = []} in
      List.iter
        (fun subset ->
          let narrowed = E.narrow subset base in
          Alcotest.(check bool)
            "budget only shrinks"
            true
            (Budget.leq narrowed.budget base.budget) ;
          List.iter
            (fun o ->
              if List.mem (E.Disable o) subset then
                Alcotest.(check bool)
                  ("disabled: " ^ optimisation_name o)
                  true
                  (List.mem o narrowed.disabled))
            all_optimisations ;
          List.iter
            (fun g ->
              Alcotest.(check bool)
                ("revoked: " ^ grant_name g)
                (List.mem (E.Revoke g) subset)
                ((not (Budget.permits narrowed.budget g))
                && Budget.permits base.budget g))
            (List.filter (fun g -> Budget.permits base.budget g) all_grants) ;
          List.iter
            (fun o ->
              List.iter
                (fun backend ->
                  if applied (decide narrowed o ~backend) then
                    Alcotest.(check bool)
                      (Printf.sprintf
                         "%s on %s applied only if it was"
                         (optimisation_name o)
                         backend)
                      true
                      (applied (decide base o ~backend)))
                backends)
            all_optimisations)
        subsets)
    [full_budget; Budget.exact]

let test_narrow_nothing () =
  let base = {budget = full_budget; disabled = []} in
  let same = E.narrow [] base in
  Alcotest.(check string)
    "no item, same policy"
    (effective_digest base)
    (effective_digest same)

let test_narrow_keeps_existing_disabled () =
  let base = {budget = full_budget; disabled = [Auto_unroll]} in
  let narrowed = E.narrow [E.Disable Launch_window] base in
  Alcotest.(check (list string))
    "union, in vocabulary order"
    ["window"; "unroll"]
    (List.map optimisation_name narrowed.disabled)

(* ------------------------------------------------------------------------ *)
(* Explain lines                                                             *)
(* ------------------------------------------------------------------------ *)

let decision ?(kernel = "k") ?(target = "d") optimisation decision =
  Decision {kernel; target; optimisation; decision}

let test_lines () =
  let check name expected event =
    Alcotest.(check string) name expected (E.explain_line event)
  in
  check
    "proof"
    "sarek-explain k d narrowing-proof applied proof=3/5 narrowings are exact"
    (decision Narrowing_proof (Applied (Proof "3/5 narrowings are exact"))) ;
  check
    "granted"
    "sarek-explain k d epilogue applied granted=cross-kernel-contraction \
     reason=\"profile fast\""
    (decision
       Epilogue_fusion
       (Applied
          (Granted
             {
               grants = [Deviation Cross_kernel_contraction];
               reason = "profile fast";
               lowering = None;
             }))) ;
  check
    "several grants"
    "sarek-explain k d matrix applied \
     granted=automatic-float-rewrite,accumulation-order reason=\"a; b\""
    (decision
       Matrix_rewrite
       (Applied
          (Granted
             {
               grants =
                 [Consent Automatic_float_rewrite; Deviation Accumulation_order];
               reason = "a; b";
               lowering = None;
             }))) ;
  check
    "exact"
    "sarek-explain k d unroll applied exact"
    (decision Auto_unroll (Applied Exact_by_construction)) ;
  check
    "refused"
    "sarek-explain k d window refused disabled (SAREK_OPT_DISABLE)"
    (decision Launch_window (Refused "disabled (SAREK_OPT_DISABLE)")) ;
  check
    "quotes in a reason are escaped, newlines flattened"
    "sarek-explain k d matrix applied granted=accumulation-order reason=\"say \
     \\\"hi\\\" now\""
    (decision
       Matrix_rewrite
       (Applied
          (Granted
             {
               grants = [Deviation Accumulation_order];
               reason = "say \"hi\"\nnow";
               lowering = None;
             }))) ;
  check
    "spaces in kernel and device names become underscores, a pair is joined"
    "sarek-explain p+c AMD_Radeon_RX epilogue applied exact"
    (decision
       ~kernel:"p + c"
       ~target:"AMD Radeon RX"
       Epilogue_fusion
       (Applied Exact_by_construction)) ;
  check
    "matrix: the lowering is a trailing field"
    "sarek-explain k HIP:0 matrix applied granted=accumulation-order \
     reason=\"profile fast\" lowering=\"f16 operands, staged\""
    (decision
       ~target:"HIP:0"
       Matrix_rewrite
       (Applied
          (Granted
             {
               grants = [Deviation Accumulation_order];
               reason = "profile fast";
               lowering = Some "f16 operands, staged";
             }))) ;
  check
    "coopmat gate has its own label"
    "sarek-explain k HIP:0 coopmat-gate refused coopmat-relaxed-accumulation"
    (Gate
       {
         kernel = "k";
         target = "HIP:0";
         decision = Refused "coopmat-relaxed-accumulation";
       }) ;
  check
    "grant on a generated name"
    "sarek-explain sarek_efused_1 - policy refused grant-on-generated-name \
     single-rounding"
    (Ignored_grant
       {
         kernel = "sarek_efused_1";
         grant = Deviation Single_rounding;
         cause = Generated_name;
       }) ;
  check
    "revoked grant"
    "sarek-explain k - policy refused grant-revoked single-rounding"
    (Ignored_grant
       {kernel = "k"; grant = Deviation Single_rounding; cause = Revoked})

let test_parse_explain () =
  let t =
    Alcotest.testable
      (fun ppf -> function
        | E.Explain_off -> Format.pp_print_string ppf "off"
        | E.Explain_stderr -> Format.pp_print_string ppf "stderr"
        | E.Explain_unknown v -> Format.fprintf ppf "unknown %S" v)
      ( = )
  in
  List.iter
    (fun (input, expected) ->
      Alcotest.check
        t
        (match input with None -> "unset" | Some s -> Printf.sprintf "%S" s)
        expected
        (E.parse_explain input))
    [
      (None, E.Explain_off);
      (Some "1", E.Explain_stderr);
      (Some "0", E.Explain_off);
      (Some "", E.Explain_off);
      (Some "true", E.Explain_unknown "true");
      (Some "json:/tmp/x", E.Explain_unknown "json:/tmp/x");
      (Some " 1", E.Explain_unknown " 1");
    ]

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "sarek_policy_env"
    [
      ( "SAREK_OPT_DISABLE parser",
        [
          case "table" test_parse_table;
          case "every name" test_parse_every_name;
          case "names are distinct" test_names_distinct;
          case "unknown item message" test_unknown_message;
        ] );
      ( "restrictions only restrict",
        [
          case "never grants (2^12 subsets)" test_narrow_never_grants;
          case "no item is the identity" test_narrow_nothing;
          case "keeps what is disabled" test_narrow_keeps_existing_disabled;
        ] );
      ( "SAREK_EXPLAIN",
        [
          case "line format" test_lines;
          case "value of the variable" test_parse_explain;
        ] );
    ]
