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
(** Mathias Bourgoin - 2014                             *)

open Spoc




let _ = 
  let devices = Spoc.Devices.init () in
  Array.iter (fun dev ->
      Printf.printf "Test on device : %s\n" dev.Devices.general_info.Devices.name;
      for i = 0 to 1024 do
        let v1 = Spoc.Vector.create Vector.float32 1024 in
        let v2 = Spoc.Vector.create Vector.float32 1024 in
        Mem.to_device v1 dev;
        Mem.to_device v2 dev;
        Mem.to_cpu v1 ();
        Devices.flush dev ();
      done; Gc.compact ();) devices
