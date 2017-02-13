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
(** Mathias Bourgoin - 2011                             *)

open Spoc

kernel vec_add : Spoc.Vector.vfloat32 -> Spoc.Vector.vfloat32 -> Spoc.Vector.vfloat32 -> int -> unit = "kernels/VecAdd" "vec_add"
kernel vec_add_double : Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> Spoc.Vector.vfloat64 -> int -> unit = "kernels/VecAdd" "vec_add_double"

let devices = Spoc.Devices.init ()

let dev = ref devices.(0)
let vec_size = ref 1024
let auto_transfers = ref false
let verify = ref true

let _ =
  Random.self_init();
  let arg1 = ("-device" , Arg.Int (fun i -> dev := devices.(i)), "number of the device [0]")
  and arg2 = ("-size" , Arg.Int (fun i -> vec_size := i), "size of the vectors to multiply [1024]")
  and arg3 = ("-auto" , Arg.Bool (fun b -> auto_transfers := b), "let Spoc handles transfers automatically [false]")
  and arg4 = ("-verify" , Arg.Bool (fun b -> verify := b), "verify computation [true]") in
  Arg.parse ([arg1; arg2; arg3; arg4]) (fun s -> ()) "";
  
  let allow_double = Spoc.Devices.allowDouble !dev in
  
  Spoc.Mem.auto_transfers !auto_transfers;
  Printf.printf "Will use device : %s\n" (!dev).Spoc.Devices.general_info.Spoc.Devices.name;
  Printf.printf "Size of vectors : %d\n" !vec_size;
  if allow_double then 
    begin
      Printf.printf "Will use double precision\n";
      Printf.printf "Allocating Vectors (on CPU memory)\n";
      flush stdout;
      
      let a = Spoc.Vector.create Spoc.Vector.float64 (!vec_size)
      and b = Spoc.Vector.create Spoc.Vector.float64 (!vec_size)
      and res = Spoc.Vector.create Spoc.Vector.float64 (!vec_size) in
      let vec_add = vec_add_double in
      Printf.printf "Loading Vectors with random floats\n";
      flush stdout;
      for i = 0 to (Spoc.Vector.length a - 1) do
	Spoc.Mem.set a i (Random.float 32.);
	Spoc.Mem.set b i ( (Random.float 32.) );
      done;
      if (not !auto_transfers) then
	begin
	  Printf.printf "Transfering Vectors (on Device memory)\n";
	  flush stdout;
	  Spoc.Mem.to_device a !dev;
	  Spoc.Mem.to_device b !dev;
	  Spoc.Mem.to_device res !dev;
	  
	(* Kernel launch : computation *)
	end;
      begin
    	Printf.printf "Computing\n";
    	flush stdout;
    	let threadsPerBlock = match !dev.Devices.specific_info with
      	  | Devices.OpenCLInfo clI ->
            (match clI.Devices.device_type with
            | Devices.CL_DEVICE_TYPE_CPU -> 1
            | _ -> 256)
      	  | _ -> 256
    	in
    	let blocksPerGrid = (!vec_size + threadsPerBlock -1) / threadsPerBlock in
    	let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
    	let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in
	
	vec_add#compile (~debug: true) !dev;
    	Spoc.Kernel.run !dev (block, grid) vec_add (a, b, res, !vec_size);
    	Pervasives.flush stdout;
      end;	
      if (not !auto_transfers) then
	begin
	  Printf.printf "Transfering Vectors (on CPU memory)\n";
	  Pervasives.flush stdout;
	  Spoc.Mem.to_cpu res ();
	end;
      Spoc.Devices.flush !dev ();
      if !verify then
	(
	  Printf.printf "Verifying Computation\n";
	  flush stdout;
	  let correct = ref true in
	  begin
            for i = 0 to (Spoc.Vector.length res - 1) do
              let tmp = (Spoc.Mem.get a i) +. (Spoc.Mem.get b i) in
              if ((tmp) -. (Spoc.Mem.get res i) > 1.e-8) then
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
    end
  else
    begin
      Printf.printf "Will use simple precision\n";
      Printf.printf "Allocating Vectors (on CPU memory)\n";
      flush stdout;
      
      let a = Spoc.Vector.create Spoc.Vector.float32 (!vec_size)
      and b = Spoc.Vector.create Spoc.Vector.float32 (!vec_size)
      and res = Spoc.Vector.create Spoc.Vector.float32 (!vec_size) in
      let vec_add = vec_add in
      Printf.printf "Loading Vectors with random floats\n";
      flush stdout;
      for i = 0 to (Spoc.Vector.length a - 1) do
    	Spoc.Mem.set a i (Random.float 32.);
    	Spoc.Mem.set b i ( (Random.float 32.) );
      done;
      if (not !auto_transfers) then
	begin
	  Printf.printf "Transfering Vectors (on Device memory)\n";
	  flush stdout;
	  Spoc.Mem.to_device a !dev;
	  Spoc.Mem.to_device b !dev;
	  Spoc.Mem.to_device res !dev;
	  
    (* Kernel launch : computation *)
	end;
      begin
    	Printf.printf "Computing\n";
    	flush stdout;
    	let threadsPerBlock = match !dev.Devices.specific_info with
      	  | Devices.OpenCLInfo clI ->
            (match clI.Devices.device_type with
            | Devices.CL_DEVICE_TYPE_CPU -> 1
            | _ -> 256)
      	  | _ -> 256
    	in
    	let blocksPerGrid = (!vec_size + threadsPerBlock -1) / threadsPerBlock in
    	let block = { Spoc.Kernel.blockX = threadsPerBlock; Spoc.Kernel.blockY = 1 ; Spoc.Kernel.blockZ = 1;} in
    	let grid = { Spoc.Kernel.gridX = blocksPerGrid; Spoc.Kernel.gridY = 1 ; Spoc.Kernel.gridZ = 1;} in
	
	vec_add#compile (~debug: true) !dev;
    	Spoc.Kernel.run !dev (block, grid) vec_add (a, b, res, !vec_size);
    	Pervasives.flush stdout;
      end;	
      if (not !auto_transfers) then
	begin
	  Printf.printf "Transfering Vectors (on CPU memory)\n";
	  Pervasives.flush stdout;
	  Spoc.Mem.to_cpu res ();
	end;
      Spoc.Devices.flush !dev ();
      if !verify then
	(
	  Printf.printf "Verifying Computation\n";
	  flush stdout;
	  let correct = ref true in
	  begin
            for i = 0 to (Spoc.Vector.length res - 1) do
              let tmp =  (Spoc.Mem.get a i) +. (Spoc.Mem.get b i) in
              if ((tmp) -. (Spoc.Mem.get res i) > 1.e-5) then
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
	)
    end;
  flush stdout;
  Printf.printf "Press any key to close\n";
  Pervasives.flush stdout;
  Spoc.Devices.closeOutput ();
  let a = read_line() in a
    
