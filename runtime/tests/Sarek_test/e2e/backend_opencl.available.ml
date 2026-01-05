(* OpenCL backend available - initialize unless disabled by env var *)
let init () =
  let dominated_by_gpu = Sys.getenv_opt "SPOC_DISABLE_GPU" = Some "1" in
  let disabled = Sys.getenv_opt "SPOC_DISABLE_OPENCL" = Some "1" in
  if not (disabled || dominated_by_gpu) then Sarek_opencl.Opencl_plugin.init ()
