(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(* Laws of Sarek_policy (specs/optimisation-policy.md).

   The Budget laws are pure and checked exhaustively over the 2^7 subsets of
   the seven grant atoms. The store is process-wide and one-way, so the cases
   about it run in order in this one executable, each on atoms and kernel names
   no earlier case granted. *)

open Sarek_policy
module B = Budget

(* ------------------------------------------------------------------------ *)
(* Budgets over every subset of the atoms                                    *)
(* ------------------------------------------------------------------------ *)

let atoms = all_grants

let subsets =
  List.fold_left (fun acc g -> acc @ List.map (fun s -> g :: s) acc) [[]] atoms

let add_exn ~reason g b =
  match B.add g ~reason b with
  | Ok b -> b
  | Error _ -> Alcotest.fail "add refused a non-blank reason"

let build ?(tag = "r") subset =
  List.fold_left
    (fun b g -> add_exn ~reason:(tag ^ grant_name g) g b)
    B.exact
    subset

let budgets = List.map build subsets

let grant_t =
  Alcotest.testable
    (fun ppf g -> Format.pp_print_string ppf (grant_name g))
    ( = )

let test_atoms () =
  Alcotest.(check int) "seven atoms" 7 (List.length atoms) ;
  Alcotest.(check int) "128 subsets" 128 (List.length budgets) ;
  Alcotest.(check (list string))
    "grant names"
    [
      "accumulation-order";
      "operand-rounding-f16";
      "operand-rounding-bf16";
      "operand-rounding-bf16x3";
      "single-rounding";
      "cross-kernel-contraction";
      "automatic-float-rewrite";
    ]
    (List.map grant_name atoms) ;
  Alcotest.(check (list string))
    "optimisation names"
    ["window"; "epilogue"; "matrix"; "narrowing-proof"; "unroll"]
    (List.map optimisation_name all_optimisations)

let test_exact_is_bottom () =
  List.iter
    (fun b -> Alcotest.(check bool) "exact <= b" true (B.leq B.exact b))
    budgets ;
  List.iter
    (fun g ->
      Alcotest.(check bool) "exact permits nothing" false (B.permits B.exact g))
    atoms ;
  Alcotest.(check (list (pair grant_t string)))
    "no grants"
    []
    (B.grants B.exact)

let test_leq_preorder () =
  List.iter
    (fun a -> Alcotest.(check bool) "reflexive" true (B.leq a a))
    budgets ;
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          List.iter
            (fun c ->
              if B.leq a b && B.leq b c then
                Alcotest.(check bool) "transitive" true (B.leq a c))
            budgets)
        budgets)
    budgets

let test_meet_is_glb () =
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          let m = B.meet a b in
          Alcotest.(check bool) "meet <= left" true (B.leq m a) ;
          Alcotest.(check bool) "meet <= right" true (B.leq m b) ;
          List.iter
            (fun c ->
              if B.leq c a && B.leq c b then
                Alcotest.(check bool) "greatest lower bound" true (B.leq c m))
            budgets)
        budgets)
    budgets

let test_meet_reason_left () =
  let a = add_exn ~reason:"left" (Deviation Single_rounding) B.exact in
  let b = add_exn ~reason:"right" (Deviation Single_rounding) B.exact in
  Alcotest.(check (option string))
    "left reason"
    (Some "left")
    (B.reason (B.meet a b) (Deviation Single_rounding)) ;
  Alcotest.(check (option string))
    "right reason"
    (Some "right")
    (B.reason (B.meet b a) (Deviation Single_rounding))

let test_permits_monotone () =
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          if B.leq a b then
            List.iter
              (fun g ->
                if B.permits a g then
                  Alcotest.(check bool) "monotone" true (B.permits b g))
              atoms)
        budgets)
    budgets

let test_digest () =
  List.iter2
    (fun subset a ->
      let reordered = build ~tag:"other " (List.rev subset) in
      Alcotest.(check string)
        "independent of reasons and insertion order"
        (B.digest a)
        (B.digest reordered))
    subsets
    budgets ;
  List.iter
    (fun a ->
      List.iter
        (fun b ->
          let same = B.leq a b && B.leq b a in
          Alcotest.(check bool)
            "digest equal iff same grant set"
            same
            (B.digest a = B.digest b))
        budgets)
    budgets

let test_add_blank () =
  List.iter
    (fun reason ->
      List.iter
        (fun g ->
          match B.add g ~reason B.exact with
          | Error Blank_reason -> ()
          | _ -> Alcotest.failf "blank reason %S accepted" reason)
        atoms)
    [""; " "; "\t\n "]

let test_first_reason_kept () =
  let g = Deviation Accumulation_order in
  let b = add_exn ~reason:"second" g (add_exn ~reason:"first" g B.exact) in
  Alcotest.(check (option string)) "first kept" (Some "first") (B.reason b g) ;
  Alcotest.(check int) "one entry" 1 (List.length (B.grants b))

let test_grants_sorted () =
  let b = build (List.rev atoms) in
  Alcotest.(check (list string))
    "stable order"
    (List.map grant_name atoms)
    (List.map (fun (g, _) -> grant_name g) (B.grants b))

let test_no_implication () =
  List.iter
    (fun g ->
      let b = add_exn ~reason:"only" g B.exact in
      List.iter
        (fun h ->
          Alcotest.(check bool)
            (grant_name g ^ " does not permit " ^ grant_name h)
            (g = h)
            (B.permits b h))
        atoms)
    atoms

(* ------------------------------------------------------------------------ *)
(* The process-wide store (run in order)                                     *)
(* ------------------------------------------------------------------------ *)

let refusal_t =
  Alcotest.testable
    (fun ppf r -> Format.pp_print_string ppf (refusal_message r))
    ( = )

let granted = Alcotest.(check (result unit refusal_t)) "granted" (Ok ())

let test_refusals () =
  Alcotest.(check (result unit refusal_t))
    "blank"
    (Error Blank_reason)
    (grant Process (Deviation Cross_kernel_contraction) ~reason:" ") ;
  Alcotest.(check (result unit refusal_t))
    "blank kernel scope"
    (Error Blank_reason)
    (grant (Kernel "kern_a") (Deviation Cross_kernel_contraction) ~reason:"") ;
  Alcotest.(check (result unit refusal_t))
    "default kernel name"
    (Error Default_kernel_name)
    (grant
       (Kernel "sarek_kern")
       (Deviation Cross_kernel_contraction)
       ~reason:"r") ;
  Alcotest.(check bool)
    "messages differ"
    true
    (refusal_message Blank_reason <> refusal_message Default_kernel_name) ;
  List.iter
    (fun g ->
      Alcotest.(check (option string))
        "refusals record nothing"
        None
        (reason ~kernel:"kern_a" g))
    [Deviation Cross_kernel_contraction]

let test_budget_is_meet () =
  let x = Deviation Accumulation_order in
  granted (grant (Kernel "kern_a") x ~reason:"a only") ;
  Alcotest.(check bool) "[] lacks it" false (B.permits (budget ~kernels:[]) x) ;
  Alcotest.(check bool)
    "[A] has it"
    true
    (B.permits (budget ~kernels:["kern_a"]) x) ;
  Alcotest.(check bool)
    "[B] lacks it"
    false
    (B.permits (budget ~kernels:["kern_b"]) x) ;
  Alcotest.(check bool)
    "[A; B] lacks it"
    false
    (B.permits (budget ~kernels:["kern_a"; "kern_b"]) x) ;
  Alcotest.(check bool)
    "[B; A] lacks it"
    false
    (B.permits (budget ~kernels:["kern_b"; "kern_a"]) x)

let test_process_grant_reaches_all () =
  let y = Deviation Single_rounding in
  granted (grant Process y ~reason:"process") ;
  List.iter
    (fun kernels ->
      Alcotest.(check bool)
        "process grant in every budget"
        true
        (B.permits (budget ~kernels) y))
    [[]; ["kern_a"]; ["kern_a"; "kern_b"]] ;
  Alcotest.(check bool)
    "kernel-scoped atom still absent for [A; B]"
    false
    (B.permits
       (budget ~kernels:["kern_a"; "kern_b"])
       (Deviation Accumulation_order))

let test_reason_precedence () =
  let z = Deviation (Operand_rounding Operands_f16) in
  granted (grant Process z ~reason:"process") ;
  granted (grant (Kernel "kern_a") z ~reason:"kernel") ;
  Alcotest.(check (option string))
    "kernel first"
    (Some "kernel")
    (reason ~kernel:"kern_a" z) ;
  Alcotest.(check (option string))
    "falls back to process"
    (Some "process")
    (reason ~kernel:"kern_b" z) ;
  Alcotest.(check (option string))
    "no ~kernel reads the process scope only"
    (Some "process")
    (reason z) ;
  let w = Deviation (Operand_rounding Operands_bf16) in
  granted (grant (Kernel "kern_a") w ~reason:"kernel only") ;
  Alcotest.(check (option string))
    "kernel-only grant, no ~kernel"
    None
    (reason w)

let test_first_grant_kept () =
  let c = Consent Automatic_float_rewrite in
  granted (grant Process c ~reason:"one") ;
  granted (grant Process c ~reason:"two") ;
  Alcotest.(check (option string)) "first kept" (Some "one") (reason c) ;
  granted (grant (Kernel "kern_c") c ~reason:"k1") ;
  granted (grant (Kernel "kern_c") c ~reason:"k2") ;
  Alcotest.(check (option string))
    "first kernel reason kept"
    (Some "k1")
    (reason ~kernel:"kern_c" c)

let test_grants_do_not_imply () =
  (* After the cases above, [Operand_rounding Operands_bf16x3] and
     [Cross_kernel_contraction] were never granted in any scope. *)
  List.iter
    (fun g ->
      Alcotest.(check (option string))
        (grant_name g ^ " never granted")
        None
        (reason ~kernel:"kern_a" g) ;
      Alcotest.(check bool)
        (grant_name g ^ " absent from the budget")
        false
        (B.permits (budget ~kernels:["kern_a"]) g))
    [
      Deviation (Operand_rounding Operands_bf16x3);
      Deviation Cross_kernel_contraction;
    ]

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run
    "sarek_policy"
    [
      ( "budget laws",
        [
          case "atoms and names" test_atoms;
          case "exact is bottom" test_exact_is_bottom;
          case "leq is a preorder" test_leq_preorder;
          case "meet is the greatest lower bound" test_meet_is_glb;
          case "meet keeps the left reason" test_meet_reason_left;
          case "permits is monotone" test_permits_monotone;
          case "digest" test_digest;
          case "add refuses a blank reason" test_add_blank;
          case "first reason kept" test_first_reason_kept;
          case "grants are sorted" test_grants_sorted;
          case "no grant implies another" test_no_implication;
        ] );
      ( "store (in order)",
        [
          case "refusals" test_refusals;
          case "budget is the meet over kernels" test_budget_is_meet;
          case
            "process grant reaches every budget"
            test_process_grant_reaches_all;
          case "reason precedence" test_reason_precedence;
          case "first grant kept per scope" test_first_grant_kept;
          case "no implication in the store" test_grants_do_not_imply;
        ] );
    ]
