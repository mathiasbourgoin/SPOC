(******************************************************************************
 * Sarek PPX - GPU kernel DSL for OCaml
 *
 * This is the PPX entry point. It registers the [%kernel ...] extension
 * and orchestrates parsing, type checking, lowering, and code generation.
 ******************************************************************************)

open Ppxlib
open Sarek_ppx_lib

(** The main kernel expansion function *)
let expand_kernel ~ctxt payload =
  let loc = Expansion_context.Extension.extension_point_loc ctxt in

  try
    (* 1. Parse the PPX payload to Sarek AST *)
    let ast = Sarek_parse.parse_payload payload in

    (* 2. Set up the typing environment with stdlib *)
    let env = Sarek_env.(empty |> with_stdlib) in

    (* 3. Type inference *)
    (match Sarek_typer.infer_kernel env ast with
     | Error errors ->
       (* Report the first error with location *)
       Sarek_error.report_errors errors;
       (* If we get here, generate dummy expression *)
       [%expr assert false]

     | Ok tkernel ->
       (* 4. Lower to Kirc_Ast *)
       let ir = Sarek_lower.lower_kernel tkernel in
       let ret_val = Sarek_lower.lower_return_value tkernel in

       (* 5. Quote the IR back to OCaml *)
       Sarek_quote.quote_kernel ~loc tkernel ir ret_val)

  with
  | Sarek_parse.Parse_error_exn (msg, ploc) ->
    Location.raise_errorf ~loc:ploc "Sarek parse error: %s" msg
  | Sarek_error.Sarek_error e ->
    let eloc = Sarek_ast.loc_to_ppxlib (Sarek_error.error_loc e) in
    Location.raise_errorf ~loc:eloc "%a" Sarek_error.pp_error e
  | e ->
    Location.raise_errorf ~loc "Sarek internal error: %s" (Printexc.to_string e)

(** The [%kernel ...] extension *)
let kernel_extension =
  Extension.V3.declare
    "kernel"
    Extension.Context.expression
    Ast_pattern.(single_expr_payload __)
    expand_kernel

(** Register the transformation *)
let () =
  Driver.register_transformation
    ~extensions:[kernel_extension]
    "sarek_ppx"
