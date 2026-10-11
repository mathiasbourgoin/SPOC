(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Compile-time advice for matrix products (specs/matrix-pattern-advice.md).
 *
 * After a kernel is lowered, every GEMM nest that Sarek_matrix_pattern
 * recognises and for which Sarek_matrix_advice knows a matrix-unit
 * configuration becomes an OCaml ALERT named [sarek_matrix], located at the
 * nest's [for] loop:
 *
 *   ((let module Sarek_advice : sig
 *        val matrix_product : unit [@@alert sarek_matrix "<text>"]
 *      end = struct let matrix_product = () end
 *     in Sarek_advice.matrix_product);
 *    <the kernel value>)
 *
 * WHY AN ALERT. A [@ocaml.ppwarning] is warning 22, which dune's dev profile
 * makes fatal (see Sarek_ppx.scan_file_for_sarek_types): an ADVICE must never
 * fail a build. An alert is printed with its location, is non-fatal in both dev
 * and release profiles (measured, OCaml 5.4), is shown by merlin, and is
 * silenced by standard OCaml means: [[@alert "-sarek_matrix"]] on the kernel
 * expression (ppxlib moves the extension node's attributes onto this
 * expansion) or [[@@alert "-sarek_matrix"]] on its binding, [[@@@alert ...]]
 * per file, [-alert -sarek_matrix] per library, [-alert @sarek_matrix] to make
 * it an error.
 *
 * SAREK_MATRIX_SCAN=<file> appends one line per kernel and per finding, for
 * corpus scans; unset, nothing is written.
 ******************************************************************************)

open Ppxlib
module Pattern = Sarek_matrix_pattern
module Advice = Sarek_matrix_advice

(* The [for] loops of the payload whose index is [name]. *)
let for_loops_named payload name =
  let finder =
    object
      inherit [Location.t list] Ast_traverse.fold as super

      method! expression e acc =
        let acc = super#expression e acc in
        match e.pexp_desc with
        | Pexp_for ({ppat_desc = Ppat_var {txt; _}; ppat_loc; _}, _, _, _, _)
          when String.equal txt name ->
            {e.pexp_loc with loc_end = ppat_loc.loc_end} :: acc
        | _ -> acc
    end
  in
  finder#expression payload []

(* The loop when the payload names exactly one loop with this index; the start
   of the kernel otherwise (a caret on the whole payload would print it all). *)
let alert_loc ~kernel_loc ~payload (g : Pattern.gemm) =
  match for_loops_named payload g.k_var.var_name with
  | [loc] -> loc
  | _ -> {kernel_loc with loc_end = kernel_loc.loc_start}

let alert_attribute ~loc text =
  let open Ast_builder.Default in
  attribute
    ~loc
    ~name:{txt = "alert"; loc}
    ~payload:
      (PStr
         [
           pstr_eval
             ~loc
             (pexp_apply
                ~loc
                (evar ~loc "sarek_matrix")
                [(Nolabel, estring ~loc text)])
             [];
         ])

let alert_expression ~loc text =
  let open Ast_builder.Default in
  let value =
    {
      (value_description
         ~loc
         ~name:{txt = "matrix_product"; loc}
         ~type_:[%type: unit]
         ~prim:[])
      with
      pval_attributes = [alert_attribute ~loc text];
    }
  in
  let signature = pmty_signature ~loc [psig_value ~loc value] in
  let structure = pmod_structure ~loc [%str let matrix_product = ()] in
  pexp_letmodule
    ~loc
    {txt = Some "Sarek_advice"; loc}
    (pmod_constraint ~loc structure signature)
    (pexp_ident
       ~loc
       {txt = Ldot (Lident "Sarek_advice", "matrix_product"); loc})

(* ---- Corpus scan --------------------------------------------------------- *)

let case_name = function
  | Advice.F16_inputs -> "f16"
  | Advice.F32_inputs -> "f32"
  | Advice.Int32_inputs -> "int32"
  | Advice.No_case -> "no-config"

let scan_line ~where ~name = function
  | Pattern.Gemm g ->
      Printf.sprintf
        "%s\t%s\tGEMM\t%s\tA=%s(%s) B=%s(%s) C=%s(%s) k=%s batch=%d"
        where
        name
        (case_name (Advice.advise g).case)
        g.a.array
        (Pattern.layout_name g.a.layout)
        g.b.array
        (Pattern.layout_name g.b.layout)
        g.c.array
        (Pattern.layout_name g.c.layout)
        g.k_var.var_name
        (List.length g.batch)
  | Pattern.Near_miss nm ->
      Printf.sprintf
        "%s\t%s\tNEAR_MISS\t%s/%s\t%s"
        where
        name
        nm.nm_acc.var_name
        nm.nm_loop.var_name
        (Pattern.reason_text nm.nm_reason)

let record_scan ~kernel_loc ~name findings =
  match Sys.getenv_opt "SAREK_MATRIX_SCAN" with
  | None | Some "" -> ()
  | Some path ->
      let where =
        Printf.sprintf
          "%s:%d"
          kernel_loc.loc_start.pos_fname
          kernel_loc.loc_start.pos_lnum
      in
      let lines =
        Printf.sprintf
          "%s\t%s\tKERNEL\tfindings=%d"
          where
          name
          (List.length findings)
        :: List.map (scan_line ~where ~name) findings
      in
      let oc = open_out_gen [Open_creat; Open_append; Open_text] 0o644 path in
      Fun.protect
        ~finally:(fun () -> close_out_noerr oc)
        (fun () -> List.iter (fun l -> output_string oc (l ^ "\n")) lines)

(* ---- Entry point --------------------------------------------------------- *)

(** [wrap ~kernel_loc ~payload ?name ir result] analyses [ir] (unless it has a
    [%native] block), records the scan when requested, and returns [result]
    preceded by one alert per advised GEMM nest ([result] itself when there is
    none). *)
let wrap ~kernel_loc ~payload ?name (ppx_ir : Sarek_ir_ppx.kernel) result =
  (* A kernel with an inline [%native] block is not analysed: the block is
     opaque (it may write any array or read the accumulator), and the
     converter refuses it. *)
  let findings =
    if Sarek_ir_conv.has_native ppx_ir.kern_body then []
    else Pattern.analyze (Sarek_ir_conv.conv_kernel ppx_ir)
  in
  record_scan
    ~kernel_loc
    ~name:(Option.value name ~default:"<anonymous>")
    findings ;
  let alerts =
    List.filter_map
      (function
        | Pattern.Gemm g ->
            let adv = Advice.advise g in
            if Advice.has_matrix_unit adv then
              let loc = alert_loc ~kernel_loc ~payload g in
              Some
                (alert_expression ~loc (Advice.render ?kernel_name:name g adv))
            else None
        | Pattern.Near_miss _ -> None)
      findings
  in
  List.fold_right
    (fun alert acc ->
      Ast_builder.Default.pexp_sequence ~loc:result.pexp_loc alert acc)
    alerts
    result
