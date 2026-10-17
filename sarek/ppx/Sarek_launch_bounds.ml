(******************************************************************************
 * Sarek PPX - Launch-bounds pragma
 *
 * [pragma ["sarek.max_threads N"]] wrapping the whole kernel body declares an
 * upper bound on the work-group size the kernel may be launched with. The
 * pragma is consumed here: lowering strips it and records the bound in the
 * kernel IR ([kern_launch_bounds]).
 ******************************************************************************)

open Sarek_typed_ast

let directive = "sarek.max_threads"

(** Largest work-group size any supported backend accepts. A bound above it can
    never be met, so it is refused as a mistake rather than carried along. *)
let max_allowed = 1024

type parsed = Not_a_launch_bound | Bound of int | Malformed of string

(** Classify pragma options. Both [["sarek.max_threads N"]] and
    [["sarek.max_threads"; "N"]] spellings are accepted, like [sarek.inline]. *)
let parse (opts : string list) : parsed =
  let words =
    List.concat_map
      (fun o ->
        String.split_on_char
          ' '
          (String.map (fun c -> if c = '\t' then ' ' else c) o)
        |> List.filter (fun w -> w <> ""))
      opts
  in
  let is_decimal w =
    w <> "" && String.for_all (fun c -> c >= '0' && c <= '9') w
  in
  let prefix_len = String.length directive in
  match words with
  | [d; n] when d = directive -> (
      match if is_decimal n then int_of_string_opt n else None with
      | Some v when v >= 1 && v <= max_allowed -> Bound v
      | _ ->
          Malformed
            (Printf.sprintf
               "%s expects an integer between 1 and %d, got %S"
               directive
               max_allowed
               n))
  | d :: _
    when String.length d >= prefix_len && String.sub d 0 prefix_len = directive
    ->
      (* A near miss ("sarek.max_threads=256", an extra option, a missing
         number) must not be mistaken for an unrelated pragma and dropped. *)
      Malformed
        (Printf.sprintf
           "%s expects exactly one integer argument (pragma [\"%s N\"])"
           directive
           directive)
  | _ -> Not_a_launch_bound

(** Strip the leading [sarek.max_threads] pragma(s) from a kernel body. Returns
    the bound (None when absent) and the remaining body. Two declarations on one
    kernel are refused. *)
let peel (body : texpr) : int option * texpr =
  let rec go acc (e : texpr) =
    match e.te with
    | TEPragma (opts, inner) -> (
        match parse opts with
        | Bound n ->
            if acc <> None then
              Ppxlib.Location.raise_errorf
                ~loc:(Sarek_ast.loc_to_ppxlib e.te_loc)
                "%s declared twice on one kernel"
                directive
            else go (Some n) inner
        | Malformed msg ->
            Ppxlib.Location.raise_errorf
              ~loc:(Sarek_ast.loc_to_ppxlib e.te_loc)
              "%s"
              msg
        | Not_a_launch_bound -> (
            (* Another pragma may sit between; look through it, keep it. *)
            match go acc inner with
            | acc', inner' when inner' != inner ->
                (acc', {e with te = TEPragma (opts, inner')})
            | acc', _ -> (acc', e)))
    | _ -> (acc, e)
  in
  go None body

(** Refuse [sarek.max_threads] anywhere but around the whole kernel body: a
    bound that no backend would see must not be silently accepted. *)
let reject_misplaced ~(loc : Ppxlib.location) (opts : string list) : unit =
  match parse opts with
  | Not_a_launch_bound -> ()
  | Bound _ | Malformed _ ->
      Ppxlib.Location.raise_errorf
        ~loc
        "%s must wrap the whole kernel body (pragma [\"%s N\"] (...)), not an \
         inner expression"
        directive
        directive
