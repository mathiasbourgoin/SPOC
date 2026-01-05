(******************************************************************************
 * Framework Cache
 *
 * Provides a persistent on-disk cache for compiled kernels (SPIR-V, PTX, etc.)
 * to reduce startup time.
 ******************************************************************************)

let cache_dir_name = "spoc"

let get_cache_dir () =
  let base_dir =
    try Sys.getenv "XDG_CACHE_HOME"
    with Not_found -> (
      try Filename.concat (Sys.getenv "HOME") ".cache"
      with Not_found -> Filename.get_temp_dir_name ())
  in
  let dir = Filename.concat base_dir cache_dir_name in
  if not (Sys.file_exists dir) then Unix.mkdir dir 0o755 ;
  dir

let compute_key ~dev_name ~driver_version ~source =
  let ctx = Digest.string (dev_name ^ driver_version ^ source) in
  Digest.to_hex ctx

let get ~key =
  let dir = get_cache_dir () in
  let filename = Filename.concat dir key in
  if Sys.file_exists filename then
    try
      let ic = open_in_bin filename in
      let len = in_channel_length ic in
      let data = really_input_string ic len in
      close_in ic ;
      Some data
    with _ -> None
  else None

let put ~key ~data =
  let dir = get_cache_dir () in
  let filename = Filename.concat dir key in
  try
    let oc = open_out_bin filename in
    output_string oc data ;
    close_out oc
  with _ -> ()
