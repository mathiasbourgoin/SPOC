(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Unit tests for the fallback of an epilogue-merged launch
 * (Launch_epilogue.merged, #459).
 *
 * When the fused kernel fails at issue time, the pair is issued as submitted.
 * That fallback must behave like the window's own issuing: every launch is
 * attempted even if an earlier one raises, the first exception is re-raised
 * after the last, and the statistics count the backend launches actually made.
 *
 * The launches are mocks that record what was issued; no device is involved.
 ******************************************************************************)

module LW = Sarek_execute.Launch_window
module LE = Sarek_execute.Launch_epilogue

let dims1 : Spoc_framework.Framework_sig.dims = {x = 1; y = 1; z = 1}

let new_window () : int LW.t =
  LW.create
    ~int32_arg:Int32.to_int
    ~arg_buf:(fun _ -> None)
    ~arg_int32:(fun _ -> None)
    ~set_pending:(fun _ -> ())
    ~guard:(fun f -> f ())

(* A mock launch that logs [name] to [events] when issued, and raises
   [Failure name] after logging when [fails]. The fused path is [fused_result]. *)
let mock_launch ~events ~name ~fails ~fused_result : int LW.launch =
  let issue () =
    events := name :: !events ;
    if fails then failwith name
  in
  {
    LW.device_id = 0;
    backend = "HIP";
    ir = {Sarek_ir_types.default_kernel with kern_name = name};
    ir_id = 0;
    shared_bytes = Some 0;
    max_grid_x = 1024;
    max_shared = 4096;
    max_arg_slots = max_int;
    args = [];
    block = dims1;
    grid = dims1;
    shared_mem = 0;
    accesses = [];
    issue;
    issue_fused =
      (fun _ _ _ ->
        events := (name ^ ":fused") :: !events ;
        fused_result ());
  }

type outcome = {
  events : string list;  (** in issue order *)
  raised : exn option;
  issued : int;  (** the window's statistics *)
}

(* Merge a producer and a consumer, submit the merged launch to a window and
   flush it. [fused_fails]: the fused kernel raises when issued. *)
let run ~fused_fails ~p_fails ~c_fails =
  let events = ref [] in
  let w = new_window () in
  let fused_result () = if fused_fails then failwith "fused" in
  let p = mock_launch ~events ~name:"p" ~fails:p_fails ~fused_result in
  let c = mock_launch ~events ~name:"c" ~fails:c_fails ~fused_result in
  let fused = {Sarek_ir_types.default_kernel with kern_name = "fused"} in
  let m =
    Mutex.protect w.LW.lock (fun () ->
        LE.merged w p c ~fused ~p_array:"a" ~shared:[] ~tc:1)
  in
  LW.submit w m ;
  let raised = match LW.flush w with () -> None | exception e -> Some e in
  {events = List.rev !events; raised; issued = (LW.stats w).issued}

let failure_of = function Some (Failure s) -> Some s | _ -> None

let check_outcome msg ~events ~raised ~issued (o : outcome) =
  Alcotest.(check (list string)) (msg ^ ": issued launches") events o.events ;
  Alcotest.(check (option string))
    (msg ^ ": exception re-raised")
    raised
    (failure_of o.raised) ;
  Alcotest.(check int) (msg ^ ": backend launches counted") issued o.issued

let test_fused_ok () =
  check_outcome
    "fused kernel ok"
    ~events:["p:fused"]
    ~raised:None
    ~issued:1
    (run ~fused_fails:false ~p_fails:false ~c_fails:false)

let test_fallback_both_ok () =
  check_outcome
    "fallback, both ok"
    ~events:["p:fused"; "p"; "c"]
    ~raised:None
    ~issued:2
    (run ~fused_fails:true ~p_fails:false ~c_fails:false)

let test_fallback_producer_raises () =
  check_outcome
    "fallback, producer raises"
    ~events:["p:fused"; "p"; "c"]
    ~raised:(Some "p")
    ~issued:1
    (run ~fused_fails:true ~p_fails:true ~c_fails:false)

let test_fallback_consumer_raises () =
  check_outcome
    "fallback, consumer raises"
    ~events:["p:fused"; "p"; "c"]
    ~raised:(Some "c")
    ~issued:1
    (run ~fused_fails:true ~p_fails:false ~c_fails:true)

let test_fallback_both_raise () =
  check_outcome
    "fallback, both raise"
    ~events:["p:fused"; "p"; "c"]
    ~raised:(Some "p")
    ~issued:0
    (run ~fused_fails:true ~p_fails:true ~c_fails:true)

let () =
  Alcotest.run
    "Launch_epilogue fallback"
    [
      ( "merged",
        [
          Alcotest.test_case "fused kernel ok" `Quick test_fused_ok;
          Alcotest.test_case "fallback, both ok" `Quick test_fallback_both_ok;
          Alcotest.test_case
            "fallback, producer raises"
            `Quick
            test_fallback_producer_raises;
          Alcotest.test_case
            "fallback, consumer raises"
            `Quick
            test_fallback_consumer_raises;
          Alcotest.test_case
            "fallback, both raise"
            `Quick
            test_fallback_both_raise;
        ] );
    ]
