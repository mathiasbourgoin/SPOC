open Spoc

ktype complex = {mutable re :  float;  mutable im : float;};;

let matmult_complex = kern cva cvb cvc n ->
  let mul = fun c d ->
    {re = c.re *. d.re -. c.im *. d.im;
     im = c.im *. d.re +. c.re *. d.im;}
  in
  let add = fun c d -> 
    {
      re = c.re +. d.re;
      im = c.im +. d.im;
    }
  in
  let open Std in
  let row = thread_idx_y + block_dim_y * block_idx_y in 
  let col = thread_idx_x + block_dim_x * block_idx_x in 
  let mutable sum = 
    {re = 0.; 
     im = 0.} in
  if row < n && col < n then
    (
      for i = 0 to n - 1 do
        sum := add sum  (mul cva.[<row*n+i>]  cvb.[<i*n+col>]);
      done;
      cvc.[<row*n+col>] <- sum 
    )
  else ()


(* let matmul_gpu = kern a b c n -> *)
(*   let open Std in *)
(*   let row = thread_idx_y + block_dim_y * block_idx_y in *)
(*   let col = thread_idx_x + block_dim_x * block_idx_x in *)
(*   let mutable sum = 0. in *)
(*   if row < n && col < n then *)
(*    (  *)
(*     for i = 0 to n - 1 do *)
(*       sum := sum +.  a.[<row*n+i>] *.  b.[<i*n+col>]; *)
(*     done; *)
(*     c.[<row*n+col>] <- sum *)
(*    ) *)
(*   else () *)


let n = 2048
let devid = try int_of_string  Sys.argv.(1) with _ -> 1



let cpt = ref 0

let tot_time = ref 0.

let measure_time f s =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "%s : time %d : %Fs\n%!" s !cpt (t1 -. t0);
  tot_time := !tot_time +.  (t1 -. t0);
  incr cpt;
  a;;

let mul = fun c d ->
  {re = c.re *. d.re -. c.im *. d.im;
   im = c.im *. d.re +. c.re *. d.im;}

let add = fun c d -> 
  {
    re = c.re +. d.re;
    im = c.im +. d.im;
  }
  
type lol = 
{mutable xre : float;
 mutable xim : float;
 mutable yre : float;
 mutable yim : float;
}

let cpu_compute xc yc zc () =  
     let sum = ref {re = 0.; im = 0.} in 
     for row = 0 to n - 1 do 
       for col = 0 to n - 1 do 
         sum := {re = 0.; im = 0.};
         for i = 0 to n - 1 do 
           let l = {
             xre = xc.(row*n+i).re;
             xim = xc.(row*n+i).im;
             yre = yc.(i*n+col).re;
             yim = yc.(i*n+col).im;
           }
           in
           sum := { re = !sum.re +. (l.xre *. l.yre -. l.xim *. l.yim);
                    im = !sum.im +.  (l.xim *. l.yre +. l.xre *. l.yim)}
                  
(* add  *)
(*                !sum  *)
(*                (mul  *)
(*                   );  *)
         done; 
         zc.(row*n+col) <- !sum
       done 
     done 
     


let _ = 
  let devs = Devices.init () in
  let dev = devs.(devid) in
  let block_size = 4 in
  Printf.printf "Dev is %s\n" devs.(devid).Devices.general_info.Devices.name;
  let x = Vector.create (Vector.Custom customComplex) (n*n)
  and y = Vector.create (Vector.Custom customComplex) (n*n)
  and z = Vector.create (Vector.Custom customComplex) (n*n)
  and xc = Array.create (n*n) {re = 0.; im = 0.}
  and yc = Array.create (n*n) {re = 0.; im = 0.}
  and zc = Array.create (n*n) {re = 0.; im = 0.}

  and blocsA = Array.init block_size
      (fun x ->
         Array.init block_size (fun _ -> Vector.create (Vector.Custom customComplex) (n/block_size*n/block_size))) 
  and blocsB = Array.init block_size 
      (fun x ->
         Array.init block_size (fun _ -> Vector.create (Vector.Custom customComplex) (n/block_size*n/block_size))) 
  and blocsC = Array.init block_size
      (fun x ->
         Array.init block_size (fun _ -> Vector.create (Vector.Custom customComplex) (n/block_size*n/block_size))) 

  in

  for i = 0 to n*n-1 do
    let a = {re = (Random.float 3.) *. 5.; 
             im =  (Random.float 3.) *. 5.}
    and b = {re = (Random.float 3.) *. 5.; 
             im = (Random.float 3.) *. 5.} in
    Mem.set x i a;
    Mem.set y i b;

   xc.(i) <- a;
    yc.(i) <- b;
  done;

  for i = 0 to block_size - 1 do
    for j = 0 to block_size -1 do
      for n = 0 to n/block_size*n/block_size -1 do
        let a = {re = (Random.float 3.) *. 5.; 
                 im =  (Random.float 3.) *. 5.}
        and b = {re = (Random.float 3.) *. 5.; 
                 im = (Random.float 3.) *. 5.} in
        Mem.set blocsA.(i).(j) n a;
        Mem.set blocsB.(i).(j) n b;
      done;
    done;
  done;

  
  let threadsPerBlock dev = match dev.Devices.specific_info with
    | Devices.OpenCLInfo clI ->
      (match clI.Devices.device_type with
       | Devices.CL_DEVICE_TYPE_CPU -> 1
       | _ -> 16)
    | _ -> 16
  in
  let blocksPerGrid dev = (n + (threadsPerBlock dev )-1) / (threadsPerBlock dev)in
  let block dev = {Spoc.Kernel.blockX = threadsPerBlock dev; 
               Spoc.Kernel.blockY = threadsPerBlock dev; 
               Spoc.Kernel.blockZ = 1;} in
  let grid dev = {Spoc.Kernel.gridX = blocksPerGrid dev; 
              Spoc.Kernel.gridY = blocksPerGrid dev; 
              Spoc.Kernel.gridZ = 1;} in

(*  Kirc.gen matmul_gpu ;*)
  Kirc.gen  matmult_complex; 
  let name = dev.Spoc.Devices.general_info.Spoc.Devices.name in
  measure_time (fun () ->
      Kirc.run matmult_complex (x,y,z,n) (block dev,grid dev) 0 dev;
      Mem.to_cpu z ();
      Devices.flush dev ();) ("GPU "^name);

  

  let blocksPerGrid dev = (n/block_size + (threadsPerBlock dev) -1) / (threadsPerBlock dev)in
  let block dev = {Spoc.Kernel.blockX = threadsPerBlock dev; 
               Spoc.Kernel.blockY = threadsPerBlock dev; 
               Spoc.Kernel.blockZ = 1;} in
  let grid dev = {Spoc.Kernel.gridX = blocksPerGrid dev; 
              Spoc.Kernel.gridY = blocksPerGrid dev; 
              Spoc.Kernel.gridZ = 1;} in


  let tasks = ref [] in
  let taskM = Mutex.create () in
    
  let multicompute () =
    for i = 0 to block_size - 1 do
      for j = 0 to block_size -1 do
        tasks := (fun (block,grid) -> 
            for k = 0 to block_size - 1  do
              Kirc.run 
                matmult_complex 
                (Mem.sub_vector blocsA.(i).(k) 0 (n*n/block_size/block_size), 
                 Mem.sub_vector blocsB.(k).(j) 0 (n*n/block_size/block_size), 
                 blocsC.(i).(j), 
                 (n/block_size)) 
                (block,grid) 0 dev;
            done;
          ):: !tasks
      done;
    done;
  in


  let rec worker i dev = 
    Mutex.lock taskM;
    match !tasks with
    | t::q -> 
      tasks := q;
      Mutex.unlock taskM;
      t (block dev, grid dev);
      Devices.flush dev ();
      worker (i+1) dev
    | [] -> 
      (*let name = dev.Spoc.Devices.general_info.Spoc.Devices.name in
      Printf.printf "\n%s did %d tasks\n%!" name i;*)
      Mutex.unlock taskM
      
  in
  multicompute();


  measure_time (fun () ->
        let t1 = Thread.create (fun () ->worker 0 devs.(0)) () 
            
        (*and t3 = Thread.create (fun () -> worker 0 devs.(2)) () *)
        in 
        Thread.join t1;
      (*Thread.join t3; *)
    ) "1 x 680";

  for i = 0 to block_size - 1 do
    for j = 0 to block_size -1 do
      for n = 0 to n/block_size*n/block_size -1 do
        let a = {re = (Random.float 3.) *. 5.; 
                 im =  (Random.float 3.) *. 5.}
        and b = {re = (Random.float 3.) *. 5.; 
                 im = (Random.float 3.) *. 5.} in
        Mem.set blocsA.(i).(j) n a;
        Mem.set blocsB.(i).(j) n b;
      done;
    done;
  done;

  multicompute();

  measure_time (fun () -> 
        
        let t1 = Thread.create (fun () ->worker 0 devs.(0)) () 
        and t2 = Thread.create (fun () -> worker 0 devs.(1)) () 
        (*and t3 = Thread.create (fun () -> worker 0 devs.(2)) () *)
        in 
        Thread.join t1;
        Thread.join t2; 
      (*Thread.join t3; *)
    ) "2x 680";

  for i = 0 to block_size - 1 do
    for j = 0 to block_size -1 do
      for n = 0 to n/block_size*n/block_size -1 do
        let a = {re = (Random.float 3.) *. 5.; 
                 im =  (Random.float 3.) *. 5.}
        and b = {re = (Random.float 3.) *. 5.; 
                 im = (Random.float 3.) *. 5.} in
        Mem.set blocsA.(i).(j) n a;
        Mem.set blocsB.(i).(j) n b;
      done;
    done;
  done;

  multicompute();

  measure_time (fun () -> 
      let t1 = Thread.create (fun () ->worker 0 devs.(0)) () 
      and t2 = Thread.create (fun () -> worker 0 devs.(1)) () 
      and t3 = Thread.create (fun () -> worker 0 devs.(2)) () 
      in 
      Thread.join t1;
      Thread.join t2; 
      Thread.join t3; 
    ) "2x 680 + i7";
    
    measure_time (cpu_compute xc yc zc) "CPU"


     
 
