(******************************************************************************
 * Sarek Framework - Plugin Registry (Stub)
 *
 * This is a placeholder for the future plugin system.
 * Full implementation pending ctypes integration.
 ******************************************************************************)

let plugins : (string, (module Framework_sig.S)) Hashtbl.t = Hashtbl.create 8

let register (module P : Framework_sig.S) =
  Hashtbl.replace plugins P.name (module P : Framework_sig.S)

let find name = Hashtbl.find_opt plugins name

let names () = Hashtbl.to_seq_keys plugins |> List.of_seq

let available () =
  Hashtbl.to_seq_values plugins
  |> Seq.filter (fun (module P : Framework_sig.S) ->
      try P.is_available () with _ -> false)
  |> List.of_seq
