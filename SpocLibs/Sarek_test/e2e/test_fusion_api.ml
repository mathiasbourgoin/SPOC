(** Integration test for Sarek.Kirc.Fusion API

    Tests kernel fusion using the runtime API with actual PPX-defined kernels.
*)

(** Producer kernel: temp[i] = input[i] * 2 *)
let producer =
  [%kernel
    fun (temp : int32 vector) (input : int32 vector) ->
      temp.(thread_idx_x) <- input.(thread_idx_x) * 2l]

(** Consumer kernel: output[i] = temp[i] + 1 *)
let consumer =
  [%kernel
    fun (output : int32 vector) (temp : int32 vector) ->
      output.(thread_idx_x) <- temp.(thread_idx_x) + 1l]

(** Test that fusion API is accessible and works *)
let test_can_fuse () =
  let _spoc_prod, kirc_prod = producer in
  let _spoc_cons, kirc_cons = consumer in
  (* Check if fusion is possible *)
  let can_fuse =
    Sarek.Kirc.Fusion.can_fuse_bodies
      kirc_prod.Sarek.Kirc.body
      kirc_cons.Sarek.Kirc.body
      ~intermediate:"temp"
  in
  Printf.printf "can_fuse: %b\n" can_fuse ;
  assert can_fuse ;
  Printf.printf "test_can_fuse: PASSED\n"

(** Test fusing kernel bodies directly *)
let test_fuse_bodies () =
  let _spoc_prod, kirc_prod = producer in
  let _spoc_cons, kirc_cons = consumer in
  let fused_body =
    Sarek.Kirc.Fusion.fuse_bodies
      kirc_prod.Sarek.Kirc.body
      kirc_cons.Sarek.Kirc.body
      ~intermediate:"temp"
  in
  (* Verify the fused body is a valid Kern *)
  (match fused_body with
  | Sarek.Kirc_Ast.Kern _ -> Printf.printf "fused_body is a Kern: OK\n"
  | _ -> failwith "Expected Kern") ;
  Printf.printf "test_fuse_bodies: PASSED\n"

(** Test fusing full kirc_kernel records *)
let test_fuse_kernels () =
  let _spoc_prod, kirc_prod = producer in
  let _spoc_cons, kirc_cons = consumer in
  let fused =
    Sarek.Kirc.Fusion.fuse_kernels kirc_prod kirc_cons ~intermediate:"temp"
  in
  (* Verify the fused kernel has a body *)
  (match fused.Sarek.Kirc.body with
  | Sarek.Kirc_Ast.Kern _ -> Printf.printf "fused kernel body is a Kern: OK\n"
  | _ -> failwith "Expected Kern in fused kernel") ;
  Printf.printf "test_fuse_kernels: PASSED\n"

(** Three-stage pipeline: a[i] = in[i]*2, b[i] = a[i]+1, out[i] = b[i]*3 *)
let stage1 =
  [%kernel
    fun (a : int32 vector) (input : int32 vector) ->
      a.(thread_idx_x) <- input.(thread_idx_x) * 2l]

let stage2 =
  [%kernel
    fun (b : int32 vector) (a : int32 vector) ->
      b.(thread_idx_x) <- a.(thread_idx_x) + 1l]

let stage3 =
  [%kernel
    fun (output : int32 vector) (b : int32 vector) ->
      output.(thread_idx_x) <- b.(thread_idx_x) * 3l]

(** Test pipeline fusion *)
let test_fuse_pipeline () =
  let _, k1 = stage1 in
  let _, k2 = stage2 in
  let _, k3 = stage3 in
  let bodies = [k1.Sarek.Kirc.body; k2.Sarek.Kirc.body; k3.Sarek.Kirc.body] in
  let fused_body, eliminated = Sarek.Kirc.Fusion.fuse_pipeline_bodies bodies in
  Printf.printf "Eliminated intermediates: %s\n" (String.concat ", " eliminated) ;
  (* Should eliminate both a and b *)
  assert (List.mem "a" eliminated) ;
  assert (List.mem "b" eliminated) ;
  (* Verify result is a Kern *)
  (match fused_body with
  | Sarek.Kirc_Ast.Kern _ -> Printf.printf "pipeline fused body is a Kern: OK\n"
  | _ -> failwith "Expected Kern") ;
  Printf.printf "test_fuse_pipeline: PASSED\n"

let () =
  Printf.printf "=== Fusion API Integration Tests ===\n" ;
  test_can_fuse () ;
  test_fuse_bodies () ;
  test_fuse_kernels () ;
  test_fuse_pipeline () ;
  Printf.printf "=== All integration tests passed! ===\n"
