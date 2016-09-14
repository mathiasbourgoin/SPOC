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
open Spoc

open Kirc

(*  fonction basique de mesure du temps *)
let measure_time s f =
  let t0 = Unix.gettimeofday () in
  let a = f () in
  let t1 = Unix.gettimeofday () in
  Printf.printf "time %s : %Fs\n%!" s (t1 -. t0);
  a;;



(* m : nombre de colonnes *)
(* n : nombre de lignes *)
let m = ref 512
and n = ref 512


(* calcul super naif sur GPU en utilisant le DSL Sarek *)
let dummy_computation = kern m_in m_out (n:int32) (m:int32) ->
  let open Std in
  (* chaque thread du GPU (dans une grille 2D) récupère son *)
  (* identifiant dans la grille, c'est fait automatiquement sur la *)
  (* carte graphique *)
  (* la grille de threads ressemble à ça :  *)
  (*   x,y -> *)
  (* 0,0 1,0 ... m-1,0   *)
  (* 0,1 1,0 ... m-1,0   *)
  (*       ...           *)
  (* 0,n 1,n ... m-1,n-1 *)
  let y = thread_idx_y + (block_idx_y * block_dim_y) in
  let x = thread_idx_x + (block_idx_x * block_dim_x) in
  (* on peut avoir lancé trop de threads donc il faut s'assurer que
     seuls les threads correspondant à un élément dans la matrice de
     sortie vont travailler pour éviter un segfault *)
  if x < m &&  y < n then
    begin
      (* calcul de la somme des éléments dans la fenêtre [O->i,0->j] de
         la matrice d'entrée *)
      (* pas de ref en Sarek mais des "mutable" *)
      let mutable tmp = 0 in
      for j = 0 to  y  do
        for i = 0 to x  do
          tmp := (tmp +
                  m_in.[<j* m + i >])
        done;
      done;
      (* écriture dans la matrice de sortie *)
      m_out.[<y*m + x>] <- (tmp mod 2l)
    end
;;


(* équivalent séquentiel (toujours super naif) pour comparer *)
let sequential_compute  m_in m_out n m =
  let (++) = Int32.add in

  (* pour chaque élément de la matrice de sortie *)
  for y = 0 to n -  1 do
    for x = 0 to m - 1 do
      let tmp : int32 ref = ref 0l in
      (* je calcule sur la fenetre de la matrice d'entrée *)
      for j = 0 to  y  do
        for i = 0 to x  do
          tmp :=  !tmp ++
                  (Mem.unsafe_get m_in (j* m + i))
        done;
      done;
      (* j'écris dans la matrice de sortie *)
      (* les unsafe permettent d'aller un poil plus vite en évitant à
         SPOC de vérifier automatiquement si les vecteurs sont bien en
         mémoire CPU *)
      Mem.unsafe_set m_out (y*m + x)  (Int32.rem !tmp 2l)
    done;
  done

(* affichage de matrice *)
  let print_mat  ?maxx:(mx= -1) ?maxy:(my= -1) mat  =
    let mx = if mx = - 1 then
        !m else mx
    and my = if my = - 1 then !n else my in
    for y = 0 to my - 1 do
      for x = 0 to mx - 1 do
        Printf.printf "%ld"
          (Mem.get mat (y * !m + x))
      done;
      print_newline();
    done

(* lancement du calcul sur GPU *)
let gpu_compute m_in m_out n m dev =
  let threadsPerBlock =
    let open Devices in 
    (* on va lancer des blocs 2D de 1 thread sur CPU et de 16*16
       threads sur GPU *)
    match dev.specific_info with
    | OpenCLInfo {device_type = CL_DEVICE_TYPE_CPU} -> 1
    | _  -> 16
  in
  (* définition des blocs et de la grille de blocs, de manière à avoir
     1 thread par élément de la matrice de sortie *)
  let blocksPerGridx =
    (m + threadsPerBlock -1) / threadsPerBlock
  and blocksPerGridy =
    (n + threadsPerBlock -1) / threadsPerBlock
  in
  let block = {Spoc.Kernel.blockX = threadsPerBlock;
	       Spoc.Kernel.blockY = threadsPerBlock;
               Spoc.Kernel.blockZ = 1}
  and grid = {Spoc.Kernel.gridX = blocksPerGridx;
	      Spoc.Kernel.gridY = blocksPerGridy;
              Spoc.Kernel.gridZ = 1} in

  (* on génère le code CUDA/OpenCL à partir de dummy_computation *)
  ignore(Kirc.gen ~only:Devices.OpenCL dummy_computation);
  (* on lance tout ça sur une carte graphique *)
  Kirc.run  dummy_computation (m_in, m_out, n, m) (block, grid) 0 dev;

  (* on s'assure que le calcul est fini et que la matrice de sortie
     est de retour sur le CPU *)
  (* on évitera sans doute de faire ça dans le programme final *)
  Mem.to_cpu m_out ();
  Devices.flush dev ()

(* fonction principale ! *)
let _ =
  (* on peut choisir la taille de la matrice d'entrée (et donc de sortie) via par exemple
     ./Brt.asm -n 512 -m 512 
  *)
  let arg1 = ("-n" , Arg.Int (fun i  -> n := i),
	      "N [512]") 
  and arg2 = ("-m" , Arg.Int (fun i  -> m := i),
	      "M [512]") in

  Arg.parse [arg1] (fun s -> ()) "";

  (* on initialise les carte graphiques *)
  let devs = Devices.init () in
  let dev = devs.(0) in

  (* on créé les matrices d'entrée et de sortie *)
  (* pour l'instant ce sont des vecteurs d'entiers mais on va passer a
     des bits dès que j'ai le temps de rajouter ça dans SPOC *)
  let m_in = Vector.create Vector.int32 (!n * !m) in
  let m_out = Vector.create Vector.int32 (!n * !m) in

  (* on remplie la matrice d'entrée et on s'assure que celle de sortie
     est bien "vide" *)
  for i = 0 to !n - 1 do
    for j = 0 to !m - 1  do
      Mem.unsafe_set m_in (i * !m + j) (Random.int32 2l);
      Mem.unsafe_set m_out (i * !m + j) 0l
    done;
  done;

  (* on peut afficher si on veut *)
  (*  print_mat m_in; *)

  (* on lance le calcul (et on mesure le temps de calcul) pour chaque GPU sur le PC *)
  Array.iter (fun dev -> measure_time (Printf.sprintf "GPU : %s" dev.Devices.general_info.Devices.name) (fun () -> gpu_compute  m_in m_out !n !m dev)) devs ;

  (* on lance le calcul séquentiel *)
  measure_time "CPU 1 Core" (fun () -> sequential_compute  m_in m_out !n !m) ;

  (* si on veut affiche la matrice de sortie *)
  (* print_newline(); print_newline(); print_newline();
     print_newline(); print_mat m_out;*)


  ;;
  
(* 
   pour info sur mon pc portable la sortie de ce programme est : 

   time GPU : Intel(R) Core(TM) i7-4710MQ CPU @ 2.50GHz : 2.16568281174s
   time GPU : Intel(R) HD Graphics Haswell GT2 Mobile : 1.89525196075s
   time CPU 1 Core : 266.087690115s
   

   conclusion : sur cet exemple super simpliste, j'utilise Mon CPU
   multicoeur avec OpenCL comme si c'etait une carte graphique, puis mon
   GPU intégré (pas tres rapide) puis mon CPU en séquentiel et on voit
   une différence énorme (soit il y a un bug dans le code, soit c'est
   super!!, je teste ça un peu mieux d'ici la fin de semaine et je le
   teste sur un plus gros GPU pour voir, parceque pourquoi pas!
*)
