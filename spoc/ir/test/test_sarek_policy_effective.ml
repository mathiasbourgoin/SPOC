(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* The effective policy of Sarek_policy (specs/optimisation-policy.md, M2):
   [requires] and [decide] against an independently written table, the meet
   over the members of a fused kernel, [version], the process restrictions and
   the decision reporter.

   The store is process-wide and one-way: every case uses kernel names that no
   other case grants, and the one case that restricts runs last. *)

open Sarek_policy
module B = Budget

let contraction = Deviation Cross_kernel_contraction

let single = Deviation Single_rounding

let kernel_grant name g =
  match grant (Kernel name) g ~reason:("test " ^ name) with
  | Ok () -> ()
  | Error _ -> Alcotest.fail ("grant refused for " ^ name)

let permits_for name g = B.permits (budget ~kernels:[name]) g

(* ------------------------------------------------------------------------ *)
(* The tables, written out by hand (not derived from the implementation)     *)
(* ------------------------------------------------------------------------ *)

let backends =
  [
    "HIP";
    "Metal";
    "CUDA/PTX";
    "CUDA/C";
    "CUDA";
    "OpenCL";
    "Vulkan";
    "Native";
    "Interpreter";
    "WebGPU";
    "NoSuchBackend";
  ]

(* What an optimisation needs on a backend; [None] = not on that backend. *)
let expected_requires opt backend =
  match opt with
  | Epilogue_fusion -> (
      match backend with
      | "HIP" | "Metal" -> Some []
      | "CUDA/PTX" | "CUDA/C" | "CUDA" | "OpenCL" | "Vulkan" ->
          Some [contraction]
      | _ -> None)
  | Launch_window | Matrix_rewrite | Narrowing_proof | Auto_unroll -> Some []

let grant_list =
  Alcotest.testable
    (fun ppf l ->
      Format.pp_print_string ppf (String.concat "," (List.map grant_name l)))
    ( = )

let test_requires () =
  List.iter
    (fun opt ->
      List.iter
        (fun backend ->
          Alcotest.(check (option grant_list))
            (optimisation_name opt ^ " on " ^ backend)
            (expected_requires opt backend)
            (requires opt ~backend))
        backends)
    all_optimisations

let subsets =
  List.fold_left
    (fun acc g -> acc @ List.map (fun s -> g :: s) acc)
    [[]]
    all_grants

let build subset =
  List.fold_left
    (fun b g ->
      match B.add g ~reason:("r:" ^ grant_name g) b with
      | Ok b -> b
      | Error _ -> Alcotest.fail "add refused")
    B.exact
    subset

let decision_t =
  Alcotest.testable
    (fun ppf -> function
      | Applied (Proof s) -> Format.fprintf ppf "Applied(Proof %s)" s
      | Applied (Granted {grants; reason}) ->
          Format.fprintf
            ppf
            "Applied(Granted %s: %s)"
            (String.concat "," (List.map grant_name grants))
            reason
      | Applied Exact_by_construction ->
          Format.pp_print_string ppf "Applied(Exact_by_construction)"
      | Refused s -> Format.fprintf ppf "Refused %S" s)
    ( = )

let expected_decision ~disabled subset opt backend =
  if disabled then Refused "disabled (SAREK_OPT_DISABLE)"
  else
    match expected_requires opt backend with
    | None -> Refused ("not on " ^ backend)
    | Some needed -> (
        match List.find_opt (fun g -> not (List.mem g subset)) needed with
        | Some missing ->
            Refused ("no grant " ^ grant_name missing ^ " (kernel or process)")
        | None -> (
            match needed with
            | [] -> Applied Exact_by_construction
            | g :: _ ->
                Applied
                  (Granted {grants = needed; reason = "r:" ^ grant_name g})))

(* 5 optimisations x 11 backends x restriction on/off x 128 grant subsets. *)
let test_decide_exhaustive () =
  let n = ref 0 in
  List.iter
    (fun subset ->
      let b = build subset in
      List.iter
        (fun opt ->
          List.iter
            (fun backend ->
              List.iter
                (fun disabled ->
                  incr n ;
                  let e =
                    {budget = b; disabled = (if disabled then [opt] else [])}
                  in
                  Alcotest.check
                    decision_t
                    (Printf.sprintf
                       "%s on %s, %d grants, disabled=%b"
                       (optimisation_name opt)
                       backend
                       (List.length subset)
                       disabled)
                    (expected_decision ~disabled subset opt backend)
                    (decide e opt ~backend))
                [false; true])
            backends)
        all_optimisations)
    subsets ;
  Alcotest.(check int) "cases" (128 * 5 * 11 * 2) !n

(* Restricting one optimisation leaves the others alone. *)
let test_disabled_is_per_optimisation () =
  let e = {budget = B.exact; disabled = [Auto_unroll]} in
  Alcotest.check
    decision_t
    "other optimisation unaffected"
    (Applied Exact_by_construction)
    (decide e Launch_window ~backend:"HIP")

(* ------------------------------------------------------------------------ *)
(* No grant implies another                                                  *)
(* ------------------------------------------------------------------------ *)

let test_grants_are_isolated () =
  List.iter
    (fun g ->
      let name = "iso_" ^ grant_name g in
      kernel_grant name g ;
      Alcotest.(check (list string))
        ("only " ^ grant_name g)
        [grant_name g]
        (List.map
           (fun (h, _) -> grant_name h)
           (B.grants (budget ~kernels:[name]))) ;
      List.iter
        (fun h ->
          if h <> g then
            Alcotest.(check (option string))
              (grant_name g ^ " does not give " ^ grant_name h)
              None
              (reason ~kernel:name h))
        all_grants)
    all_grants

(* ------------------------------------------------------------------------ *)
(* Fused kernels: the meet over the members                                  *)
(* ------------------------------------------------------------------------ *)

let test_fused_meet () =
  kernel_grant "fm_a" single ;
  kernel_grant "fm_b" single ;
  kernel_grant "fm_a" contraction ;
  register_fused ~name:"fm_ab" ~members:["fm_a"; "fm_b"] ;
  register_fused ~name:"fm_ac" ~members:["fm_a"; "fm_c"] ;
  Alcotest.(check bool) "both members hold it" true (permits_for "fm_ab" single) ;
  Alcotest.(check bool)
    "only one member holds it"
    false
    (permits_for "fm_ac" contraction) ;
  Alcotest.(check bool) "one member lacks it" false (permits_for "fm_ac" single) ;
  (* nested: a fused kernel inside a fused kernel *)
  register_fused ~name:"fm_nest" ~members:["fm_ab"; "fm_a"] ;
  register_fused ~name:"fm_nest_bad" ~members:["fm_ab"; "fm_c"] ;
  Alcotest.(check bool)
    "nested, all hold it"
    true
    (permits_for "fm_nest" single) ;
  Alcotest.(check bool)
    "nested, one lacks it"
    false
    (permits_for "fm_nest_bad" single) ;
  (* the fused name's own Kernel grant is not a grant of its members *)
  kernel_grant "fm_ac" single ;
  Alcotest.(check bool)
    "the fused name's own grant adds nothing"
    false
    (permits_for "fm_ac" single) ;
  (* a grant given after the fusion is registered is seen *)
  kernel_grant "fm_c" single ;
  Alcotest.(check bool) "late member grant" true (permits_for "fm_ac" single) ;
  Alcotest.(check bool)
    "unregistered kernel unchanged"
    false
    (permits_for "fm_unregistered" single) ;
  (* the budget of a launch of several kernels is the meet of their budgets *)
  Alcotest.(check bool)
    "kernels:[a; ab]"
    true
    (B.permits (budget ~kernels:["fm_a"; "fm_ab"]) single) ;
  Alcotest.(check bool)
    "kernels:[a; unrelated]"
    false
    (B.permits (budget ~kernels:["fm_a"; "fm_unregistered"]) single)

(* A malformed registration (a cycle) must not hang a launch. *)
let test_fused_cycle_terminates () =
  register_fused ~name:"cy_1" ~members:["cy_2"] ;
  register_fused ~name:"cy_2" ~members:["cy_1"] ;
  Alcotest.(check bool)
    "terminates and grants nothing"
    false
    (permits_for "cy_1" single)

(* A content-named fused kernel registered twice with different members is
   covered only by what both registrations allow. *)
let test_fused_reregistered () =
  kernel_grant "rr_a" single ;
  kernel_grant "rr_b" single ;
  register_fused ~name:"rr_f" ~members:["rr_a"; "rr_b"] ;
  Alcotest.(check bool) "first" true (permits_for "rr_f" single) ;
  register_fused ~name:"rr_f" ~members:["rr_a"; "rr_other"] ;
  Alcotest.(check bool) "second adds a member" false (permits_for "rr_f" single)

(* ------------------------------------------------------------------------ *)
(* version                                                                   *)
(* ------------------------------------------------------------------------ *)

let test_version () =
  let v0 = version () in
  kernel_grant "ver_a" single ;
  let v1 = version () in
  Alcotest.(check bool) "grant bumps" true (v1 > v0) ;
  kernel_grant "ver_a" single ;
  Alcotest.(check int) "duplicate grant is a no-op" v1 (version ()) ;
  ignore (grant (Kernel "ver_a") single ~reason:" ") ;
  ignore (grant (Kernel "sarek_kern") single ~reason:"x") ;
  Alcotest.(check int) "refusals do not bump" v1 (version ()) ;
  ignore (reason ~kernel:"ver_a" single) ;
  ignore (budget ~kernels:["ver_a"]) ;
  Alcotest.(check int) "reads do not bump" v1 (version ()) ;
  register_fused ~name:"ver_f" ~members:["ver_a"] ;
  let v2 = version () in
  Alcotest.(check bool) "register bumps" true (v2 > v1) ;
  register_fused ~name:"ver_f" ~members:["ver_a"] ;
  Alcotest.(check int) "same registration is a no-op" v2 (version ()) ;
  (match grant Process contraction ~reason:"ver process" with
  | Ok () -> ()
  | Error _ -> Alcotest.fail "process grant refused") ;
  Alcotest.(check bool) "process grant bumps" true (version () > v2)

(* ------------------------------------------------------------------------ *)
(* Reporter                                                                  *)
(* ------------------------------------------------------------------------ *)

let test_reporter () =
  let seen = ref [] in
  set_reporter
    (Some (function Decision r -> seen := r :: !seen | Ignored_grant _ -> ())) ;
  let r kernel device decision =
    {kernel; device; optimisation = Epilogue_fusion; decision}
  in
  let applied = Applied Exact_by_construction in
  report (r "k" "d" applied) ;
  report (r "k" "d" applied) ;
  Alcotest.(check int) "same outcome is reported once" 1 (List.length !seen) ;
  report (r "k" "d" (Refused "x")) ;
  Alcotest.(check int) "a new outcome is reported" 2 (List.length !seen) ;
  report (r "k" "d" (Refused "x")) ;
  Alcotest.(check int) "and only once" 2 (List.length !seen) ;
  report (r "k" "other" (Refused "x")) ;
  report (r "other" "d" (Refused "x")) ;
  report
    {
      kernel = "k";
      device = "d";
      optimisation = Matrix_rewrite;
      decision = Refused "x";
    } ;
  Alcotest.(check int)
    "key is kernel, device, optimisation"
    5
    (List.length !seen) ;
  report (r "k" "d" applied) ;
  Alcotest.(check int) "a change back is a new outcome" 6 (List.length !seen) ;
  set_reporter None ;
  report (r "k2" "d" applied) ;
  Alcotest.(check int) "no reporter, no call" 6 (List.length !seen)

(* ------------------------------------------------------------------------ *)
(* Restrictions (one-way: this case runs last)                               *)
(* ------------------------------------------------------------------------ *)

let test_restrict () =
  Alcotest.(check bool) "free at first" false (restricted Auto_unroll) ;
  Alcotest.(check (list string))
    "nothing disabled"
    []
    (List.map optimisation_name (effective ~kernel:"re_a").disabled) ;
  let v0 = version () in
  restrict Auto_unroll ;
  Alcotest.(check bool) "restricted" true (restricted Auto_unroll) ;
  Alcotest.(check bool) "version bumped" true (version () > v0) ;
  let v1 = version () in
  restrict Auto_unroll ;
  Alcotest.(check int) "restricting twice is a no-op" v1 (version ()) ;
  Alcotest.(check bool) "others free" false (restricted Launch_window) ;
  Alcotest.(check (list string))
    "effective lists it"
    ["unroll"]
    (List.map optimisation_name (effective ~kernel:"re_a").disabled) ;
  Alcotest.check
    decision_t
    "decide through effective"
    (Refused "disabled (SAREK_OPT_DISABLE)")
    (decide (effective ~kernel:"re_a") Auto_unroll ~backend:"HIP") ;
  Alcotest.check
    decision_t
    "other optimisation through effective"
    (Applied Exact_by_construction)
    (decide (effective ~kernel:"re_a") Launch_window ~backend:"HIP")

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "sarek_policy_effective"
    [
      ( "tables",
        [
          case "requires" test_requires;
          case "decide, exhaustive" test_decide_exhaustive;
          case "disabled is per optimisation" test_disabled_is_per_optimisation;
        ] );
      ( "store (in order)",
        [
          case "no grant implies another" test_grants_are_isolated;
          case "fused meet" test_fused_meet;
          case "fused cycle terminates" test_fused_cycle_terminates;
          case "fused re-registered" test_fused_reregistered;
          case "version" test_version;
          case "reporter" test_reporter;
          case "restrict" test_restrict;
        ] );
    ]
