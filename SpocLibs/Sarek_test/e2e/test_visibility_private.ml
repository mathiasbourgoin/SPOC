open Spoc

module Ast = Sarek_ppx_lib.Sarek_ast

type float32 = float

let () =
  let kernel =
    [%kernel
      fun (xs : float32 vector) (ys : float32 vector) (dst : float32 vector)
          (n : int32) ->
        let tid = thread_idx_x + (block_idx_x * block_dim_x) in
        if tid < n then
          dst.(tid) <- Visibility_lib.public_add xs.(tid) ys.(tid)]
  in

  let _, kirc_kernel = kernel in
  print_endline "=== Visibility kernel IR ===" ;
  Sarek.Kirc.print_ast kirc_kernel.Sarek.Kirc.body ;
  print_endline "===========================" ;

  let names_from_blobs blobs =
    List.filter_map
      (fun blob ->
        try
          match (Marshal.from_string blob 0 : Ast.module_item) with
          | Ast.MFun (n, _, _) -> Some n
          | Ast.MConst (n, _, _) -> Some n
        with _ -> None)
      blobs
  in
  let blobs = Spoc.Sarek_metadata.get_module_blobs () in
  let names = names_from_blobs blobs in
  if not (List.mem "public_add" names) then
    (Printf.printf "Expected public_add to be registered, saw: %s\n%!"
       (String.concat ", " names) ;
     exit 1) ;
  if List.mem "private_scale" names then
    (Printf.printf "private_scale unexpectedly registered (names: %s)\n%!"
       (String.concat ", " names) ;
     exit 1) ;
  print_endline "Visibility metadata check PASSED"
