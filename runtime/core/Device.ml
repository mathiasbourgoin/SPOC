(******************************************************************************
 * Sarek Runtime - Unified Device Abstraction
 *
 * Provides a unified view of all available compute devices across backends.
 * Queries registered plugins and presents a single array of devices.
 ******************************************************************************)

open Sarek_framework

(** Unified device representation *)
type t = {
  id : int;  (** Global device ID (0, 1, 2...) *)
  backend_id : int;  (** ID within the backend (0, 1...) *)
  name : string;  (** Human-readable device name *)
  framework : string;  (** Backend name: "CUDA", "OpenCL", "Native" *)
  capabilities : Framework_sig.capabilities;
}

(** Device initialization state *)
let devices : t array ref = ref [||]

let initialized = ref false

(** Initialize all available backends and enumerate devices *)
let init ?(frameworks = ["CUDA"; "OpenCL"]) () =
  if !initialized then !devices
  else begin
    let all_devices = ref [] in
    let global_id = ref 0 in

    frameworks
    |> List.iter (fun fw_name ->
        match Framework_registry.find_backend fw_name with
        | None -> () (* Backend not registered *)
        | Some (module B : Framework_sig.BACKEND) ->
            if B.is_available () then begin
              try
                B.Device.init () ;
                let count = B.Device.count () in
                for i = 0 to count - 1 do
                  let dev = B.Device.get i in
                  let device =
                    {
                      id = !global_id;
                      backend_id = i;
                      name = B.Device.name dev;
                      framework = B.name;
                      capabilities = B.Device.capabilities dev;
                    }
                  in
                  all_devices := device :: !all_devices ;
                  incr global_id
                done
              with _ -> () (* Skip backends that fail to initialize *)
            end) ;

    devices := Array.of_list (List.rev !all_devices) ;
    initialized := true ;
    !devices
  end

(** Get all initialized devices *)
let all () = if not !initialized then init () else !devices

(** Get device count *)
let count () = Array.length (all ())

(** Get device by global ID *)
let get id =
  let devs = all () in
  if id >= 0 && id < Array.length devs then Some devs.(id) else None

(** Get first available device (if any) *)
let first () =
  let devs = all () in
  if Array.length devs > 0 then Some devs.(0) else None

(** Filter devices by framework *)
let by_framework framework =
  all () |> Array.to_list
  |> List.filter (fun d -> d.framework = framework)
  |> Array.of_list

(** Filter devices by capability *)
let with_fp64 () =
  all () |> Array.to_list
  |> List.filter (fun d -> d.capabilities.supports_fp64)
  |> Array.of_list

(** Get the best device (first CUDA, then OpenCL, then Native) *)
let best () =
  let cuda = by_framework "CUDA" in
  if Array.length cuda > 0 then Some cuda.(0)
  else
    let opencl = by_framework "OpenCL" in
    if Array.length opencl > 0 then Some opencl.(0) else first ()

(** Reset initialization state (for testing) *)
let reset () =
  devices := [||] ;
  initialized := false

(** Pretty-print device info *)
let to_string d =
  Printf.sprintf
    "[%d] %s (%s) - %s, %.1f GB, %d MPs"
    d.id
    d.name
    d.framework
    (let major, minor = d.capabilities.compute_capability in
     if major > 0 then Printf.sprintf "SM %d.%d" major minor else d.framework)
    (Int64.to_float d.capabilities.total_global_mem
    /. (1024.0 *. 1024.0 *. 1024.0))
    d.capabilities.multiprocessor_count

(** Print all devices *)
let print_all () = all () |> Array.iter (fun d -> print_endline (to_string d))
