open Spoc
open Sarek
open Kirc


let piKern = [%kernel fun (rX : float32 vector) (rY : float32 vector) (inside : int32 vector) (n : int32) ->
  let open Std in
  let open Math in
  let open Float32 in
  let i = thread_idx_x + block_idx_x * block_dim_x in
  if i < n then begin
    let r = sqrt ( add (mul rX.%[i] rX.%[i])
                     (mul rY.%[i] rY.%[i])) in
    if (r <=. 1.) then
      [%native fun dev ->
        match dev.Spoc.Devices.specific_info with
        | Spoc.Devices.OpenCLInfo _ -> "atomic_inc (inside)"
        | _ -> "atomicAdd (inside,1)"
      ]
  end
]



let cpt = ref 0

let tot_time = ref 0.

let measure_time f s iter =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : %Fs  average : %Fs \n%!" s
    (t1 -. t0) ((t1 -. t0)/. (float_of_int iter));
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;

let devices = Devices.init ()

let size = ref 30720
let dev_id = ref 0
let block_size = ref 256

let usage () =
  Printf.printf "Usage: %s [options]\n" Sys.argv.(0);
  Printf.printf "Options:\n";
  Printf.printf "  -d <id>     Device ID (default: 0)\n";
  Printf.printf "  -s <size>   Vector size (default: 30720)\n";
  Printf.printf "  -b <size>   Block/work-group size (default: 256)\n";
  Printf.printf "  -h          Show this help\n";
  exit 0

let parse_args () =
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    match Sys.argv.(!i) with
    | "-d" -> incr i; dev_id := int_of_string Sys.argv.(!i)
    | "-s" -> incr i; size := int_of_string Sys.argv.(!i)
    | "-b" -> incr i; block_size := int_of_string Sys.argv.(!i)
    | "-h" | "--help" -> usage ()
    | arg ->
        (* Legacy: first positional arg is device id *)
        (try dev_id := int_of_string arg with _ -> ())
    ;
    incr i
  done

let _ =
  if Array.length devices = 0 then (
    Printf.printf "No GPU device found!\n";
    exit 1
  );

  parse_args ();

  (* List all available devices *)
  Printf.printf "Available devices:\n";
  Array.iteri (fun i d ->
    Printf.printf "  [%d] %s\n" i d.Devices.general_info.Devices.name
  ) devices;

  let dev = devices.(!dev_id) in
  Printf.printf "Using device: %s\n%!" dev.Devices.general_info.Devices.name;
  Printf.printf "Configuration: size=%d, block_size=%d\n%!" !size !block_size;
  Printf.printf "  -> blocks=%d, total_threads=%d\n%!"
    ((!size + !block_size - 1) / !block_size)
    (((!size + !block_size - 1) / !block_size) * !block_size);

  let vX = Vector.create Vector.float32 !size
  and vY =  Vector.create Vector.float32 !size
  and inside = Vector.create Vector.int32 1 in

  Mem.set inside 0 0l;

  measure_time (fun () ->
      ignore(Kirc.gen piKern dev);
    ) (Printf.sprintf "Time to generate kernel for \"%s\""
         dev.Devices.general_info.Devices.name) 1;

  for i = 0 to !size - 1 do
    Mem.set vX i (Random.float 1.);
    Mem.set vY i (Random.float 1.);
  done;

  let make_bg = fun dev size ->
    let threadsPerBlock = match dev.Devices.specific_info with
      | Devices.OpenCLInfo clI ->
        (match clI.Devices.device_type with
         | Devices.CL_DEVICE_TYPE_CPU -> 1
         | _  -> !block_size)
      | _  -> !block_size in
    let blocksPerGrid =
      (size + threadsPerBlock -1) / threadsPerBlock
    in

    let block0 = {Spoc.Kernel.blockX = threadsPerBlock;
		  Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1}
    and grid0= {Spoc.Kernel.gridX = blocksPerGrid;
	        Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1} in
    block0,grid0
  in

  measure_time (fun () ->
      Kirc.run piKern (vX, vY, inside, !size) (make_bg dev !size) 0 dev;
      Devices.flush dev ();
    ) (Printf.sprintf "Pi on %s" dev.Devices.general_info.Devices.name) 1;

  let pi = (float (Int32.to_int (Mem.get inside 0) * 4)) /. (float !size) in
  Printf.printf "PI = %.10g\n" pi;

  ()
;;
