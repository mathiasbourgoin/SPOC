(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The explain stream of Sarek_policy (specs/optimisation-policy.md, M3):
   distinct decisions, grants that have no effect, the join of reasons. The
   store is process-wide and one-way: every case uses kernel names that no other
   case grants. *)

open Sarek_policy

(* ------------------------------------------------------------------------ *)
(* Events about grants that have no effect                                   *)
(* ------------------------------------------------------------------------ *)

let single = Deviation Single_rounding

let contraction = Deviation Cross_kernel_contraction

let collect f =
  let seen = ref [] in
  set_reporter (Some (fun e -> seen := e :: !seen)) ;
  f () ;
  set_reporter None ;
  List.rev !seen

let record kernel =
  {
    kernel;
    target = "d";
    optimisation = Narrowing_proof;
    decision = Applied Exact_by_construction;
  }

let ignored_of events =
  List.filter_map
    (function
      | Ignored_grant {kernel; grant; cause} ->
          Some (kernel, grant_name grant, cause)
      | Decision _ | Gate _ -> None)
    events

let cause_t =
  Alcotest.testable
    (fun ppf c ->
      Format.pp_print_string
        ppf
        (match c with Generated_name -> "generated" | Revoked -> "revoked"))
    ( = )

let triple_t = Alcotest.(triple string string cause_t)

let test_generated_name_grant () =
  (match grant (Kernel "env_fused_a") single ~reason:"on a generated name" with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused") ;
  register_fused ~name:"env_fused_a" ~members:["env_member_a"] ;
  let events =
    collect (fun () ->
        report (record "env_member_a") ;
        report (record "env_fused_a") ;
        report {(record "env_fused_a") with target = "other"} ;
        report (record "env_member_a + env_fused_a"))
  in
  Alcotest.(check (list triple_t))
    "one event, at the first decision that sees the kernel, never repeated"
    [("env_fused_a", "single-rounding", Generated_name)]
    (ignored_of events) ;
  Alcotest.(check bool)
    "the decisions themselves are still reported"
    true
    (List.length events >= 4)

let test_ordinary_kernel_grant_is_not_ignored () =
  (match
     grant (Kernel "env_plain_a") contraction ~reason:"named by the program"
   with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "grant refused") ;
  let events = collect (fun () -> report (record "env_plain_a")) in
  Alcotest.(check (list triple_t)) "nothing ignored" [] (ignored_of events)

(* Distinct decisions, not changes: a decision that flips back and forth is
   printed once per distinct outcome. *)
let test_alternating_decisions_print_twice () =
  let decisions r =
    List.filter_map
      (function Decision d -> Some d | Gate _ | Ignored_grant _ -> None)
      r
  in
  let events =
    collect (fun () ->
        for _ = 1 to 3 do
          report (record "env_flip_a") ;
          report {(record "env_flip_a") with decision = Refused "window is off"}
        done)
  in
  Alcotest.(check int)
    "two distinct decisions, two events"
    2
    (List.length (decisions events))

(* A grant given after the decisions of its kernel settled is still reported,
   once, at the next report for that kernel. *)
let test_late_ignored_grant () =
  let events =
    collect (fun () ->
        report (record "env_late_f") ;
        (match grant (Kernel "env_late_f") single ~reason:"late" with
        | Ok () -> ()
        | Error _ -> Alcotest.fail "grant refused") ;
        register_fused ~name:"env_late_f" ~members:["env_late_a"] ;
        (* the same decision again: not new, but the grant is reported *)
        report (record "env_late_f") ;
        report (record "env_late_f"))
  in
  Alcotest.(check (list triple_t))
    "reported once"
    [("env_late_f", "single-rounding", Generated_name)]
    (ignored_of events) ;
  let late =
    collect (fun () ->
        report (record "env_late_a") ;
        (match grant (Kernel "env_late_g") single ~reason:"late" with
        | Ok () -> ()
        | Error _ -> Alcotest.fail "grant refused") ;
        register_fused ~name:"env_late_g" ~members:["env_late_a"] ;
        report (record "env_late_g") ;
        (* settled: the same decision again, after a new generated grant *)
        (match grant (Kernel "env_late_h") single ~reason:"late" with
        | Ok () -> ()
        | Error _ -> Alcotest.fail "grant refused") ;
        register_fused ~name:"env_late_h" ~members:["env_late_a"] ;
        report (record "env_late_h + env_late_g"))
  in
  Alcotest.(check bool)
    "a grant on a kernel whose decisions settled is reported at its next report"
    true
    (List.exists (fun (k, _, _) -> k = "env_late_h") (ignored_of late))

let test_join_reasons () =
  Alcotest.(check string)
    "de-duplicated in order"
    "a; b; c"
    (join_reasons ["a"; "b"; "a"; "c"; "b"]) ;
  Alcotest.(check string) "empty" "" (join_reasons [])

let test_seen_set_is_bounded () =
  let events =
    collect (fun () ->
        for i = 1 to 5000 do
          report
            {(record "env_cap_a") with decision = Refused (string_of_int i)}
        done ;
        (* the first decision again: the set was reset, so it is printed again *)
        report {(record "env_cap_a") with decision = Refused "1"})
  in
  let ones =
    List.filter
      (function
        | Decision {decision = Refused "1"; _} -> true
        | Decision _ | Gate _ | Ignored_grant _ -> false)
      events
  in
  Alcotest.(check int)
    "bounded memory: the set resets after its cap, and forgets"
    2
    (List.length ones)

(* With no reporter, report and report_gate return before building the key:
   nothing is allocated, whatever the size of the decision text. *)
let test_no_reporter_costs_nothing () =
  set_reporter None ;
  let r =
    {(record "env_hot_a") with decision = Refused (String.make 500 'x')}
  in
  let before = Gc.minor_words () in
  for _ = 1 to 10_000 do
    report r ;
    report_gate ~kernel:"env_hot_a" ~target:"HIP:0" r.decision
  done ;
  let words = Gc.minor_words () -. before in
  Alcotest.(check bool)
    (Printf.sprintf "no allocation without a reporter (%.0f words)" words)
    true
    (words < 1000.)

let test_no_reporter_no_event () =
  set_reporter None ;
  report (record "env_fused_a")

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "sarek_policy_events"
    [
      ( "grants without effect",
        [
          case "grant on a generated name" test_generated_name_grant;
          case
            "an ordinary kernel grant is not ignored"
            test_ordinary_kernel_grant_is_not_ignored;
          case "no reporter" test_no_reporter_no_event;
          case "late ignored grant" test_late_ignored_grant;
        ] );
      ( "distinct decisions",
        [
          case "alternating decisions" test_alternating_decisions_print_twice;
          case "join reasons" test_join_reasons;
          case "seen set is bounded" test_seen_set_is_bounded;
          case "no reporter, no cost" test_no_reporter_costs_nothing;
        ] );
    ]
