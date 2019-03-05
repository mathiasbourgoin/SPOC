(*
         DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                    Version 2, December 2004 

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net> 

 Everyone is permitted to copy and distribute verbatim or modified 
 copies of this license document, and changing it is allowed as long 
 as the name is changed. 

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

  0. You just DO WHAT THE FUCK YOU WANT TO.
*)
(** VecAdd.ml                                           *)
(** Add 2 vectors allocated from OCaml                  *)
(** Use prewritten kernel accessible from Spoc          *)
(**                                                     *)
(** Mathias Bourgoin - 2011 - 2019                      *)
open Spoc

kernel vecadd: Spoc.Vector.vfloat32 -> Spoc.Vector.vfloat32 -> Spoc.Vector.vfloat32 -> int -> unit = "kernels/VecAdd_MultiGPU" "vec_add"

let devices = Spoc.Devices.init ()
let dev2n = 1 mod 1 mod Array.length devices
in
let dev1 = ref devices.(0)
and dev2 = ref devices.(dev2n)

in
let start () = if ((Spoc.Devices.gpgpu_devices ()) <2 ) then
    begin
      Printf.printf "Only one compatible device found\n";
      dev2 := devices.(0)
    end
  else
    begin
      Printf.printf "Wow %d compatible devices found\n" (Spoc.Devices.gpgpu_devices ());
      dev2 := devices.(dev2n)
    end

and vec_size = ref 1024
and auto_transfers = ref false
and verify = ref true

in
let compute () =
  start();
  Random.self_init();
  let arg0 = ("-device1" , Arg.Int (fun i -> dev1 := devices.(i)), "number of the device [0]")
  and arg1 = ("-device2" , Arg.Int (fun i -> dev2 := devices.(i)), "number of the device [1 mod number_of_compatible_devices]")
  and arg2 = ("-size" , Arg.Int (fun i -> vec_size := i), "size of the vectors to multiply [1024]")
  and arg3 = ("-auto" , Arg.Bool (fun b -> auto_transfers := b; ), "let Spoc handles transfers automatically [false]")
  and arg4 = ("-verify" , Arg.Bool (fun b -> verify := b), "verify computation [true]") in
  Arg.parse ([arg0; arg1; arg2; arg3; arg4]) (fun s -> ()) "";
  Printf.printf "Will use devices : %s and %s\n" (!dev1).Spoc.Devices.general_info.Spoc.Devices.name (!dev2).Spoc.Devices.general_info.Spoc.Devices.name ;
  Printf.printf "Size of vectors : %d\n" !vec_size;
  Printf.printf "Allocating Vectors (on CPU memory)\n";
  Printf.printf "Set auto-transfers %b\n" !auto_transfers;
  Spoc.Mem.auto_transfers !auto_transfers;
  flush stdout;
  let a = Spoc.Vector.create Spoc.Vector.float32 (!vec_size)
  and b = Spoc.Vector.create Spoc.Vector.float32 (!vec_size)
  and res = Spoc.Vector.create Spoc.Vector.float32 (!vec_size) in
  let a1 = Spoc.Mem.sub_vector a 0 (!vec_size /2)
  and b1 = Spoc.Mem.sub_vector b 0 (!vec_size /2)
  and res1 = Spoc.Mem.sub_vector res 0 (!vec_size /2)
  and a2 = Spoc.Mem.sub_vector a (!vec_size /2) (!vec_size /2)
  and b2 = Spoc.Mem.sub_vector b (!vec_size /2) (!vec_size /2)
  and res2 = Spoc.Mem.sub_vector res (!vec_size /2) (!vec_size /2)
  in
  Printf.printf "Loading Vectors with random floats\n";
  flush stdout;
  for i = 0 to (Spoc.Vector.length a) - 1 do
    Spoc.Mem.set a i (Random.float 32.);
    Spoc.Mem.set b i ( (Random.float 32.)) ;
  done;
  if (not !auto_transfers) then
    begin
      Printf.printf "Transfering Vectors (on Device memory)\n";
      flush stdout;
      Spoc.Mem.to_device a1 !dev1;
      Spoc.Mem.to_device b1 !dev1;
      Spoc.Mem.to_device res1 !dev1;
      Spoc.Mem.to_device a2 !dev2;
      Spoc.Mem.to_device b2 !dev2;
      Spoc.Mem.to_device res2 !dev2;
      
      (* Kernel launch : computation *)
    end;
  begin
    Printf.printf "Computing\n";
    flush stdout;
    let threadsPerBlock dev = match dev.Devices.specific_info with
      | Devices.OpenCLInfo clI ->
          (match clI.Devices.device_type with
            | Devices.CL_DEVICE_TYPE_CPU -> 1
            | _ -> 256)
      | _ -> 256 in
    let blocksPerGrid dev = (!vec_size + (threadsPerBlock dev) -1) / (threadsPerBlock dev) in
    let block dev = { Spoc.Kernel.blockX = threadsPerBlock dev ; Spoc.Kernel.blockY = 1; Spoc.Kernel.blockZ = 1 }
    and grid dev = { Spoc.Kernel.gridX = (blocksPerGrid dev) /2; Spoc.Kernel.gridY = 1; Spoc.Kernel.gridZ = 1 } in
    vecadd#compile ~debug: true !dev1;
    vecadd#compile ~debug: true !dev2;
    Printf.printf "Computing 1st part\n";
    flush stdout;
    vecadd#run (a1, b1, res1, (!vec_size /2)) (block !dev1, grid !dev1) 0 !dev1;
    Printf.printf "Computing 2nd part\n";
    flush stdout;
    vecadd#run (a2, b2, res2, (!vec_size /2)) (block !dev2, grid !dev2) 0 !dev2;
    flush stdout;
  end;
  Printf.printf "Transfering Subvectors Back(on CPU memory)\n";
  Spoc.Mem.to_cpu res1 ();
  Spoc.Mem.to_cpu res2 ();
  Spoc.Devices.flush !dev1 ();
  Spoc.Devices.flush !dev2 ();
  if !verify then
    (
      Printf.printf "Verifying Computation\n";
      flush stdout;
      let correct = ref true in
      begin
        for i = 0 to ((Spoc.Vector.length res) - 1) do
          let tmp = a.[<i>] +. b.[<i>] in
          if ( (Vector.float32_of_float tmp) -. res.[<i>] <> 0.) then
            begin
              Printf.printf "ERROR Index: %d - %g <> %g ----> %g\n" i tmp (Spoc.Mem.get res i) (tmp -. (Spoc.Mem.get res i)) ;
              Printf.printf "a[%d] = %g, b[%d] = %g\n" i (Spoc.Mem.get a i) i (Spoc.Mem.get b i);
              correct := false;
              flush stdout
            end;
        done;
        if !correct then
          Printf.printf "Verif OK\n"
        else
          Printf.printf "Verif KO\n"
      end
    );
  flush stdout;
  let a = read_line() in
  a
in
compute()
