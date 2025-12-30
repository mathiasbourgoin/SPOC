(******************************************************************************
 * Sarek Framework - Plugin Registry
 *
 * Central registry for GPU backend plugins. Plugins register themselves
 * on module load and can be queried by name or capability.
 ******************************************************************************)

(** Registered plugins - keyed by name *)
let plugins : (string, (module Framework_sig.S)) Hashtbl.t = Hashtbl.create 8

(** Full backend plugins with complete functionality *)
let backends : (string, (module Framework_sig.BACKEND)) Hashtbl.t = Hashtbl.create 8

(** Priority ordering for auto-selection (higher = preferred) *)
let priorities : (string, int) Hashtbl.t = Hashtbl.create 8

(** Register a minimal plugin (name, version, is_available) *)
let register ?(priority = 50) (module P : Framework_sig.S) =
  Hashtbl.replace plugins P.name (module P : Framework_sig.S) ;
  Hashtbl.replace priorities P.name priority

(** Register a full backend plugin *)
let register_backend ?(priority = 50) (module B : Framework_sig.BACKEND) =
  Hashtbl.replace backends B.name (module B : Framework_sig.BACKEND) ;
  Hashtbl.replace plugins B.name (module B : Framework_sig.S) ;
  Hashtbl.replace priorities B.name priority

(** Find a plugin by name *)
let find name = Hashtbl.find_opt plugins name

(** Find a full backend by name *)
let find_backend name = Hashtbl.find_opt backends name

(** List all registered plugin names *)
let names () = Hashtbl.to_seq_keys plugins |> List.of_seq

(** List all available plugins (those where is_available returns true) *)
let available () =
  Hashtbl.to_seq_values plugins
  |> Seq.filter (fun (module P : Framework_sig.S) ->
      try P.is_available () with _ -> false)
  |> List.of_seq

(** List all available full backends *)
let available_backends () =
  Hashtbl.to_seq_values backends
  |> Seq.filter (fun (module B : Framework_sig.BACKEND) ->
      try B.is_available () with _ -> false)
  |> List.of_seq

(** Get the best available backend (highest priority) *)
let best_backend () =
  let available =
    Hashtbl.to_seq backends
    |> Seq.filter (fun (_name, (module B : Framework_sig.BACKEND)) ->
        try B.is_available () with _ -> false)
    |> List.of_seq
  in
  match available with
  | [] -> None
  | backends ->
      let sorted =
        List.sort
          (fun (name1, _) (name2, _) ->
            let p1 = Hashtbl.find_opt priorities name1 |> Option.value ~default:50 in
            let p2 = Hashtbl.find_opt priorities name2 |> Option.value ~default:50 in
            compare p2 p1) (* descending *)
          backends
      in
      Some (snd (List.hd sorted))

(** Get plugin priority *)
let priority name = Hashtbl.find_opt priorities name |> Option.value ~default:50
