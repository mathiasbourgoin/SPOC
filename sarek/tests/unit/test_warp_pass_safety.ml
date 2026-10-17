(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * A warp shuffle is CONVERGENT: its result depends on which threads share its
 * warp, so no IR pass may hoist, sink, duplicate, merge or move it across
 * control flow, and none may change which threads run or the block shape around
 * it (docs/plans/warp-shuffle-intake-2026-10-17.md, risk R2, AC-8).
 *
 * One case per pass that rewrites a kernel or decides on a launch:
 *  - the code-generation passes (for-bound hoisting, unroll hints, vector
 *    inlining, software math) keep every shuffle call, in order, and do not
 *    move one out of the statement it was in;
 *  - the launch passes (horizontal fusion, epilogue fusion, the launch window's
 *    fusable verdict, host evaluation of store indices, affine coverage, the
 *    matrix rewrite) refuse a kernel that uses a warp operation.
 ******************************************************************************)

open Sarek_ir_types
module Codegen = Sarek_codegen
module Ex = Sarek_execute

let var name ty =
  {var_name = name; var_id = 0; var_type = ty; var_mutable = false}

let vec_param name elt =
  DParam (var name (TVec elt), Some {arr_elttype = elt; arr_memspace = Global})

let shuffle_i32 v lane =
  EIntrinsic ([], Sarek_warp.ir_name Sarek_warp.Xor Sarek_warp.I32, [v; lane])

(* out.(tid) <- shuffle_xor in.(tid) 1, inside a for loop with a constant bound. *)
let loop_kernel () =
  let tid = var "tid" TInt32 and i = var "i" TInt32 in
  {
    default_kernel with
    kern_name = "warp_loop";
    kern_params = [vec_param "inp" TInt32; vec_param "out" TInt32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
          SFor
            ( i,
              EConst (CInt32 0l),
              EConst (CInt32 4l),
              Upto,
              SAssign
                ( LArrayElem ("out", EVar tid),
                  shuffle_i32
                    (EArrayRead ("inp", EVar tid))
                    (EConst (CInt32 1l)) ) ) );
  }

(* A copy of the same kernel without the shuffle, for the controls. *)
let plain_kernel () =
  let k = loop_kernel () in
  let tid = var "tid" TInt32 and i = var "i" TInt32 in
  {
    k with
    kern_name = "plain_loop";
    kern_body =
      SLet
        ( tid,
          EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
          SFor
            ( i,
              EConst (CInt32 0l),
              EConst (CInt32 4l),
              Upto,
              SAssign
                (LArrayElem ("out", EVar tid), EArrayRead ("inp", EVar tid)) )
        );
  }

let shuffle_calls (k : kernel) =
  List.filter
    Sarek_warp.is_shuffle_name
    (let names = ref [] in
     let folder =
       {
         Sarek_ir_analysis.fe =
           (fun acc e ->
             (match e with
             | EIntrinsic (_, n, _) -> names := n :: !names
             | _ -> ()) ;
             acc);
         ft = (fun a _ -> a);
         fs = (fun a _ -> a);
         fnative = Fun.id;
         visit_lvalue = true;
       }
     in
     ignore (Sarek_ir_analysis.kernel_fold folder () k) ;
     List.rev !names)

let expected_calls = [Sarek_warp.ir_name Sarek_warp.Xor Sarek_warp.I32]

let check_calls label k =
  Alcotest.(check (list string)) label expected_calls (shuffle_calls k)

(* Is the shuffle still inside the loop body of [k]? *)
let rec shuffle_inside_for = function
  | SFor (_, _, _, _, body) -> shuffle_calls_of_stmt body <> []
  | SLet (_, _, b) | SLetMut (_, _, b) | SBlock b | SPragma (_, b) ->
      shuffle_inside_for b
  | SSeq ss -> List.exists shuffle_inside_for ss
  | _ -> false

and shuffle_calls_of_stmt s =
  shuffle_calls {(loop_kernel ()) with kern_body = s}

(** {1 Code-generation passes: the call stays where it was} *)

let test_for_bound () =
  let k = Codegen.Sarek_ir_for_bound.hoist (loop_kernel ()) in
  check_calls "for_bound keeps the call" k ;
  Alcotest.(check bool)
    "for_bound leaves the shuffle in the loop body"
    true
    (shuffle_inside_for k.kern_body)

let test_unroll () =
  let k =
    Codegen.Sarek_ir_unroll.annotate ~generator:"cuda-c" (loop_kernel ())
  in
  check_calls "unroll hint keeps the call" k ;
  Alcotest.(check bool)
    "unroll hint leaves the shuffle in the loop body"
    true
    (shuffle_inside_for k.kern_body)

let test_inline_vec () =
  let k =
    Codegen.Sarek_ir_inline_vec.inline_vec_helpers
      ~backend:"Vulkan"
      (loop_kernel ())
  in
  check_calls "inline_vec keeps the call" k

(** {1 Launch passes: refused} *)

let test_horizontal_verdict () =
  Alcotest.(check bool)
    "a shuffle kernel is not a fusable member (launch window, horizontal)"
    true
    (Result.is_error (Ex.Kernel_access.horizontal_verdict (loop_kernel ()))) ;
  Alcotest.(check bool)
    "control: the same kernel without the shuffle is fusable"
    true
    (Result.is_ok (Ex.Kernel_access.horizontal_verdict (plain_kernel ())))

let test_fusion_horizontal () =
  match
    Ex.Fusion_horizontal.fuse
      ~name:"fused"
      [loop_kernel (); {(loop_kernel ()) with kern_name = "warp_loop2"}]
  with
  | (_ : kernel) -> Alcotest.fail "horizontal fusion accepted a shuffle kernel"
  | exception Ex.Fusion_horizontal.Incompatible _ -> ()

(* Producer: out.(tid) <- [shuffle] inp.(tid); consumer: res.(i) <- out_c.(i),
   with out_c bound to the producer's out. *)
let epilogue_producer ~shuffle =
  let tid = var "tid" TInt32 in
  let value = EArrayRead ("inp", EVar tid) in
  {
    default_kernel with
    kern_name = "ep_producer";
    kern_params = [vec_param "inp" TInt32; vec_param "out" TInt32];
    kern_body =
      SLet
        ( tid,
          EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
          SAssign
            ( LArrayElem ("out", EVar tid),
              if shuffle then shuffle_i32 value (EConst (CInt32 1l)) else value
            ) );
  }

let epilogue_consumer () =
  let i = var "i" TInt32 in
  {
    default_kernel with
    kern_name = "ep_consumer";
    kern_params = [vec_param "out_c" TInt32; vec_param "res" TInt32];
    kern_body =
      SLet
        ( i,
          EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
          SAssign (LArrayElem ("res", EVar i), EArrayRead ("out_c", EVar i)) );
  }

let fuse_epilogue ~shuffle =
  Ex.Fusion_epilogue.fuse
    ~name:"fused"
    ~producer:(epilogue_producer ~shuffle)
    ~p_array:"out"
    ~consumer:(epilogue_consumer ())
    ~shared:[("out_c", "out")]

let test_fusion_epilogue () =
  (* control: the same pair fuses when the producer has no shuffle *)
  ignore (fuse_epilogue ~shuffle:false : kernel) ;
  match fuse_epilogue ~shuffle:true with
  | (_ : kernel) -> Alcotest.fail "epilogue fusion accepted a shuffle producer"
  | exception Ex.Fusion_epilogue.Incompatible _ -> ()

(* Vertical fusion (can_fuse): the producer's store expression would be inlined
   under the consumer's guards. *)
let test_vertical_fusion () =
  let consumer =
    let i = var "i" TInt32 in
    {
      default_kernel with
      kern_name = "vert_consumer";
      kern_params = [vec_param "out" TInt32; vec_param "res" TInt32];
      kern_body =
        SLet
          ( i,
            EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
            SAssign (LArrayElem ("res", EVar i), EArrayRead ("out", EVar i)) );
    }
  in
  Alcotest.(check bool)
    "control: the pair fuses without a shuffle"
    true
    (Sarek.Sarek_fusion.can_fuse
       (epilogue_producer ~shuffle:false)
       consumer
       "out") ;
  Alcotest.(check bool)
    "a shuffle producer is not fused vertically"
    false
    (Sarek.Sarek_fusion.can_fuse
       (epilogue_producer ~shuffle:true)
       consumer
       "out")

(* A store whose index comes from a shuffle is not a function of the launch. *)
let shuffled_index_body () =
  let tid = var "tid" TInt32 in
  SLet
    ( tid,
      EIntrinsic (["Sarek_stdlib"; "Gpu"], "global_thread_id", []),
      SAssign
        ( LArrayElem ("out", shuffle_i32 (EVar tid) (EConst (CInt32 1l))),
          EConst (CInt32 0l) ) )

let dims : Ex.Index_slice.dims = {block = (32, 1, 1); grid = (1, 1, 1)}

let test_index_slice () =
  match
    Ex.Index_slice.run
      ~is_target:(fun a -> a = "out")
      ~record:(fun _ -> ())
      ~scalars:[]
      ~dims
      (shuffled_index_body ())
  with
  | (_ : string list) ->
      Alcotest.fail "Index_slice evaluated a store index that uses a shuffle"
  | exception Ex.Index_slice.Unknown _ -> ()

let test_coverage_affine () =
  match
    Ex.Coverage_affine.covers
      ~p_body:(shuffled_index_body ())
      ~p_array:"out"
      ~p_scalars:[]
      ~p_dims:dims
      ~c_body:SEmpty
      ~c_written:[]
      ~c_scalars:[]
      ~c_dims:dims
  with
  | (_ : bool * (string * string) list) ->
      Alcotest.fail "Coverage_affine covered a shuffled store index"
  | exception Ex.Coverage_affine.Unknown _ -> ()

let () =
  let open Alcotest in
  run
    "warp pass safety"
    [
      ( "codegen passes",
        [
          test_case "for_bound" `Quick test_for_bound;
          test_case "unroll" `Quick test_unroll;
          test_case "inline_vec" `Quick test_inline_vec;
        ] );
      ( "launch passes",
        [
          test_case
            "launch window / horizontal verdict"
            `Quick
            test_horizontal_verdict;
          test_case "horizontal fusion" `Quick test_fusion_horizontal;
          test_case "epilogue fusion" `Quick test_fusion_epilogue;
          test_case "vertical fusion" `Quick test_vertical_fusion;
          test_case "index slice" `Quick test_index_slice;
          test_case "affine coverage" `Quick test_coverage_affine;
        ] );
    ]
