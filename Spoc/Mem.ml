(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 * 
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability. 
 * 
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or 
 * data to be ensured and,  more generally, to use and operate it in the 
 * same conditions as regards security. 
 * 
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
 *******************************************************************************)
open Cuda

open OpenCL



let auto = ref true

let unsafe = ref false


let auto_transfers b = auto := b

let unsafe_rw b = unsafe := b

let unsafe_set vect idx value =
  Vector.unsafe_set vect idx value

let unsafe_get vect idx =
  Vector.unsafe_get vect idx

let rec flush_and_transfer_to_cpu d vect =
  Devices.flush d  ();
  to_cpu vect ();
  Devices.flush d ()

and set vect (idx : int) (value : 'a) =
  if !unsafe then unsafe_set vect idx value
  else
    (if (idx < 0) || (idx >= Vector.length vect)
     then raise (Invalid_argument "index out of bounds");
     if !auto
     then
       (match Vector.dev vect with
        | Vector.Dev d  | Vector.Transferring d -> 
          (flush_and_transfer_to_cpu d vect)
        | Vector.No_dev -> ());
     match Vector.is_sub vect with
     | Some (1, start, ok_r, 0, v2) -> unsafe_set v2 (start + idx) value
     | Some (1, start, ok_r, ko_r, v2) ->
       unsafe_set v2
         ((start + ((idx / ok_r) * (ko_r + ok_r))) + (idx mod ok_r)) value
     | Some (_, start, ok_r, ko_r, v2) ->
       set v2 ((start + ((idx / ok_r) * (ko_r + ok_r))) + (idx mod ok_r))
         value
     | None -> unsafe_set vect idx value)
    
and get vect idx =
  if !unsafe then unsafe_get vect idx 
  else
  (if (idx < 0) || (idx >= Vector.length vect)
   then raise (Invalid_argument "index out of bounds");
   if !auto
   then
     (match Vector.dev vect with
      | Vector.Dev d | Vector.Transferring d ->
        (flush_and_transfer_to_cpu d vect)
      | Vector.No_dev -> ());
   match Vector.is_sub vect with
   | Some (1, start, ok_r, 0, v2) -> unsafe_get v2 (start + idx)
   | Some (1, start, ok_r, ko_r, v2) ->
     unsafe_get v2
       ((start + ((idx / ok_r) * (ko_r + ok_r))) + (idx mod ok_r))
   | Some (_, start, ok_r, ko_r, v2) ->
     get v2 ((start + ((idx / ok_r) * (ko_r + ok_r))) + (idx mod ok_r))
   | None -> unsafe_get vect idx)

and temp_vector vect =
  Vector.temp_vector vect

and basic_transfer_to_device new_vect q dev =
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ ->
     let f () =
       (match Vector.vector new_vect with
        | Vector.CustomArray custom ->
          (cuda_custom_cpu_to_device new_vect dev.Devices.general_info.Devices.id
             dev.Devices.general_info q;)
        | _ ->
          (cuda_cpu_to_device new_vect dev.Devices.general_info.Devices.id
             dev.Devices.general_info dev.Devices.gc_info q;)
       )
     in
     (try f ()
      with
      | Cuda.ERROR_OUT_OF_MEMORY ->
        (try (Devices.flush dev (); f ())
         with | _ -> (Gc.full_major (); f ())))
   | Devices.OpenCLInfo _ ->
     let f () =
       (match Vector.vector new_vect with
        | Vector.CustomArray custom ->
          (opencl_custom_cpu_to_device new_vect
             (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
             dev.Devices.general_info q;)
        | _ ->
          (opencl_cpu_to_device new_vect
             (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
             dev.Devices.general_info q;)
       )
     in
     (try f ()
      with
      | _ ->
        (try (Devices.flush dev (); f ())
         with | _ -> (Gc.full_major (); f ())))
  );

and basic_transfer_to_cpu new_vect q dev =
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ ->
     let f () =
       (match Vector.vector new_vect with
        | Vector.CustomArray custom ->
          (cuda_custom_device_to_cpu new_vect dev.Devices.general_info.Devices.id
             dev.Devices.general_info q;)
        | _ ->
          (cuda_device_to_cpu new_vect dev.Devices.general_info.Devices.id
             dev.Devices.general_info dev q;)
       )
     in
     (try f ()
      with
      | Cuda.ERROR_OUT_OF_MEMORY ->
        (try (Devices.flush dev (); f ())
         with | _ -> (Gc.full_major (); f ())))
   | Devices.OpenCLInfo _ ->
     let f () =
       (match Vector.vector new_vect with
        | Vector.CustomArray custom ->
          (opencl_custom_device_to_cpu new_vect
             (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
             dev.Devices.general_info dev.Devices.specific_info q;)
        | _ ->
          (opencl_device_to_cpu new_vect
             (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
             dev.Devices.general_info dev.Devices.specific_info q;)
       )
     in
     (try f ()
      with
      | _ ->
        (try (Devices.flush dev (); f ())
         with | _ -> (Gc.full_major (); f ())))
  );

and transfer_part_to_device vect sub_vect q (dev : Devices.device) host_offset guest_offset start size =
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ -> 
     (match Vector.vector vect with
      | Vector.CustomArray custom -> 
        (cuda_custom_part_cpu_to_device vect sub_vect dev.Devices.general_info.Devices.id
           dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
      | _ -> (cuda_part_cpu_to_device vect sub_vect dev.Devices.general_info.Devices.id
                dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
     )     
   | Devices.OpenCLInfo _ -> 
     (match Vector.vector vect with
      | Vector.CustomArray custom -> (opencl_custom_part_cpu_to_device vect sub_vect
                                        (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
                                        dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
      | _ -> (opencl_part_cpu_to_device vect sub_vect
                (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
                dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
     ) 
  )

and transfer_part_to_cpu vect sub_vect q (dev : Devices.device) host_offset guest_offset start size =
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ -> 
     (match Vector.vector vect with
      | Vector.CustomArray custom -> 
        (cuda_custom_part_device_to_cpu vect sub_vect dev.Devices.general_info.Devices.id
           dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
      | _ -> (cuda_part_device_to_cpu vect sub_vect dev.Devices.general_info.Devices.id
                dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
     )     
   | Devices.OpenCLInfo _ -> 
     (match Vector.vector vect with
      | Vector.CustomArray custom -> (opencl_custom_part_device_to_cpu vect sub_vect
                                        (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
                                        dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
      | _ -> (opencl_part_device_to_cpu vect sub_vect
                (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))
                dev.Devices.general_info dev.Devices.gc_info q host_offset guest_offset start size;)
     ) 
  )

and to_device vect ?queue_id:(q = 0) (dev : Devices.device) =	
  (* if vector is a sub_vector then *)
  (*  if sub_vector depth = 1 then *)
  (*   sub_vector parts are contiguous *)
  (*   if sub_vector_parts size (ok_range) > "some correct size" for transfers then *)
  (*     transfer each ok range directly *)
  (*   else *)
  (*     if sub_vector size  < 2 * "some correct size" then *)
  (*       copy sub_vector into one contiguous vector and transfer it *)
  (*     else *)
  (*       (while subvector not entirely transfered do *)
  (*          copy subvector into one contiguous vector of "some correct size" size; *)
  (*          transfer this vector in asynchronously; *)
  match Vector.dev vect with
  | Vector.Transferring d  -> 
    ( if d <> dev then
        (Devices.flush d ();
         to_device vect ~queue_id: q dev))
  | Vector.Dev d ->
    (if d <> dev then
       (to_cpu vect ~queue_id: q ();
        Devices.flush d ();
        to_device vect ~queue_id: q dev))
  | Vector.No_dev ->
    (Vector.set_device vect (dev.Devices.general_info.Devices.id) (Vector.Transferring dev);
     (try alloc_vect_on_device vect dev with
      | Cuda.ERROR_OUT_OF_MEMORY | OpenCL.MEM_OBJECT_ALLOCATION_FAILURE | _ ->
        (try (Devices.flush dev ~queue_id:q (); alloc_vect_on_device vect dev)
         with 
         | Cuda.ERROR_OUT_OF_MEMORY | OpenCL.MEM_OBJECT_ALLOCATION_FAILURE -> 
           ( Devices.flush dev (); Gc.compact ();
             alloc_vect_on_device vect dev)
         | e -> raise e));
     match Vector.is_sub vect with
     | None -> basic_transfer_to_device vect q dev
     | Some (1, _start, _ok, _ko, v)  -> 
       if _ok = 0 then
         (
           transfer_part_to_device v vect q dev 0 0 _start (Vector.length vect) ;
         )
       else
       if _ok > 512 then
         (
           let to_transfer = ref (Vector.length vect) in
           let cpt = ref 0 in
           while !to_transfer  > _ok do
             transfer_part_to_device v vect q dev (!cpt * (_ok + _ko)) (!cpt*_ok)_start _ok ;
             to_transfer := !to_transfer - _ok;
             incr cpt;
           done;
           if !to_transfer > 0 then
             transfer_part_to_device v vect q dev (!cpt * (_ok + _ko)) (!cpt*_ok) _start !to_transfer;
         )
       else
         (
           let to_transfer = ref (Vector.length vect) in
           let cpt = ref 0 in
           let i = ref 0 in
           while !to_transfer  > 512 do
             let temp = Vector.create (Vector.kind vect) 512 in
             Vector.copy_sub vect temp;
             for idx = !i to (!i + 511) do
               set temp idx (get vect !i);
             done;
             i := !i + 512;
             transfer_part_to_device v temp q dev (!cpt * (512 + _ko)) (!cpt*512) _start 512 ;
             to_transfer := !to_transfer - 512;
             incr cpt;
           done;
           if !to_transfer > 0 then
             (
               let temp = Vector.create (Vector.kind vect) !to_transfer in
               for idx = !i to (!i + !to_transfer - 1) do
                 set temp idx (get vect !i);
               done;
               Vector.copy_sub vect temp;
               transfer_part_to_device v temp q dev (!cpt * (512 + _ko)) (!cpt*512) _start !to_transfer;
             )
         )			
     | Some (_, _, _, _, v) -> 
       (	let new_vect = temp_vector vect in
         for i = 0 to Vector.length vect - 1 do 
           unsafe_set new_vect i (get vect i)
         done;
         basic_transfer_to_device new_vect q dev;
         Vector.update_device_array new_vect vect;
       ));  
    Vector.set_device vect (dev.Devices.general_info.Devices.id) (Vector.Dev dev)

and free_vect_on_device vector dev =
  match dev.Devices.specific_info with
  | Devices.CudaInfo _ -> cuda_free_vect vector dev.Devices.general_info.Devices.id
  | Devices.OpenCLInfo _  -> opencl_free_vect vector  (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))

and alloc_vect_on_device vector dev =
  match dev.Devices.specific_info with
  | Devices.CudaInfo _ -> 
    (match Vector.vector vector with
     | Vector.Bigarray _  -> cuda_alloc_vect vector dev.Devices.general_info.Devices.id dev.Devices.general_info
     | Vector.CustomArray _  -> cuda_custom_alloc_vect vector dev.Devices.general_info.Devices.id dev.Devices.general_info)
  | Devices.OpenCLInfo _ -> 
    (match Vector.vector vector with
     | Vector.Bigarray _  ->   opencl_alloc_vect vector  (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())) dev.Devices.general_info
     | Vector.CustomArray _  -> opencl_custom_alloc_vect vector (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())) dev.Devices.general_info)

and to_cpu vect ?queue_id:(q = 0) () =
  match Vector.dev vect with
  | Vector.Transferring d  -> 
    (Devices.flush d ();
     to_cpu vect ~queue_id:q ())     
  | Vector.No_dev -> ()
  | Vector.Dev dev ->
    Vector.set_device vect (dev.Devices.general_info.Devices.id) (Vector.Transferring dev);
    (match Vector.is_sub vect with
     | None -> basic_transfer_to_cpu vect q dev
     | Some (1, _start, _ok, _ko, v)  -> 
       if _ok = 0 then
         (
           transfer_part_to_cpu v vect q dev 0 0 _start (Vector.length vect) ;
         )
       else
       if _ok > 512 then
         (
           let to_transfer = ref (Vector.length vect) in
           let cpt = ref 0 in
           while !to_transfer  > _ok do
             transfer_part_to_cpu v vect q dev (!cpt * (_ok + _ko)) (!cpt*_ok)_start _ok ;
             to_transfer := !to_transfer - _ok;
             incr cpt;
           done;
           if !to_transfer > 0 then
             transfer_part_to_cpu v vect q dev (!cpt * (_ok + _ko)) (!cpt*_ok)_start !to_transfer ;
         )
       else
         (
           let to_transfer = ref (Vector.length vect) in
           let cpt = ref 0 in
           let i = ref 0 in
           while !to_transfer  > 512 do
             let temp = Vector.create (Vector.kind vect) 512 in
             Vector.copy_sub vect temp;
             for idx = !i to (!i + 511) do
               set temp idx (get vect !i);
             done;
             transfer_part_to_cpu v temp q dev (!cpt * (512 + _ko)) (!cpt*512) _start 512 ;
             to_transfer := !to_transfer - 512;
             incr cpt;
           done;
           if !to_transfer > 0 then
             (
               let temp = Vector.create (Vector.kind vect) !to_transfer in
               for idx = !i to (!i + !to_transfer - 1) do
                 set temp idx (get vect !i);
               done;
               Vector.copy_sub vect temp;
               transfer_part_to_cpu v temp q dev (!cpt * (512 + _ko)) (!cpt*512) _start !to_transfer;
             )
         )
     | Some (_, _, _, _, v) -> 
       let new_vect = temp_vector vect
       in 
       (
         Vector.update_device_array new_vect vect;
         basic_transfer_to_cpu new_vect q dev;
         for i = 0 to (Vector.length new_vect) - 1 do
           set vect i (unsafe_get new_vect i)
         done			
       ));
    Vector.set_device vect (dev.Devices.general_info.Devices.id) (Vector.No_dev)





let sub_vector (vect : ('a, 'b) Vector.vector) _start ?ok_rng:(_ok_r = 0)
    ?ko_rng:(_ko_r = 0) _len =
  if (_ok_r > _len) || ((_ok_r < 0) || (_ko_r < 0))
  then failwith "Wrong value: sub_vector"
  else
    (if (_start + _len) > Vector.length vect
     then raise (Invalid_argument "index out of bounds")
     else
       (match Vector.dev vect with
        | Vector.Dev d | Vector.Transferring d ->
          (Devices.flush d ~queue_id: 0 ();
           to_cpu vect ~queue_id: 0 ();
           Devices.flush d ~queue_id: 0 ())
        | Vector.No_dev -> ());
     incr Vector.vec_id;
     Vector.sub_vector vect _start _ok_r _ko_r _len)


let gpu_vector_copy vecA startA vecB startB size dev = 
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ -> 
     (match Vector.vector vecA with
      | Vector.CustomArray custom -> (
          Cuda.cuda_custom_vector_copy vecA startA vecB startB size dev.Devices.general_info dev.Devices.general_info.Devices.id)
      | _ -> (
          Cuda.cuda_vector_copy vecA startA vecB startB size dev.Devices.general_info dev.Devices.general_info.Devices.id))
   | Devices.OpenCLInfo _ -> 
     (match Vector.vector vecA with
      | Vector.CustomArray custom ->  (OpenCL.opencl_custom_vector_copy vecA startA vecB startB size dev.Devices.general_info (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())))
      | _ -> (OpenCL.opencl_vector_copy vecA startA vecB startB size dev.Devices.general_info (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))))
  )

let cpu_vector_copy vecA startA vecB startB size = 
  let sB = ref startB in
  for i = startA to startA + size - 1 do
    set vecB !sB (get vecA i);
    incr sB;
  done

(* copy vector vecA data into vecB (takes place into vecA location) *)   
let rec vector_copy (vecA: ('a,'b) Vector.vector) startA (vecB: ('a,'b) Vector.vector) startB size =
  if (startA + size) > Vector.length vecA || (startB + size) > Vector.length vecB then
    raise (Invalid_argument "index out of bounds" ) 
  else
    match Vector.dev vecA, Vector.dev vecB with
    | Vector.Dev d1, Vector.Dev d2 ->
      if d1 = d2 then
        gpu_vector_copy vecA startA vecB startB size d1
      else
        (Devices.flush d2 ();
         to_device vecB d1;
         Devices.flush d1 ();
         gpu_vector_copy vecA startA vecB startB size d1 )
    |  Vector.Dev d1 , Vector.Transferring d2 ->
      if d1 = d2 then
        (Devices.flush d1 ();
         gpu_vector_copy vecA startA vecB startB size d1)
      else
        (Devices.flush d2 ();
         to_device vecB d1;
         Devices.flush d1 ();
         gpu_vector_copy vecA startA vecB startB size d1)
    | Vector.Dev d1, Vector.No_dev ->
      (to_device vecB d1;
       Devices.flush d1 ();
       gpu_vector_copy vecA startA vecB startB size d1) 
    | Vector.Transferring d1 ,  _ ->
      Devices.flush d1 ();
      vector_copy vecA startA vecB startB size;
    | Vector.No_dev, Vector.No_dev ->
      cpu_vector_copy vecA startA vecB startB size
    | Vector.No_dev, Vector.Dev d2 
    | Vector.No_dev, Vector.Transferring d2 ->
      to_cpu vecB ();
      Devices.flush d2 ();
      cpu_vector_copy vecA startA vecB startB size

let gpu_matrix_copy (vecA: ('a,'b) Vector.vector) ldA start_rowA start_colA (vecB: ('a,'b) Vector.vector) ldB start_rowB start_colB rows cols dev =
  (match dev.Devices.specific_info with
   | Devices.CudaInfo _ -> 
     (match Vector.vector vecA with
      | Vector.CustomArray custom -> (
          Cuda.cuda_custom_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols dev.Devices.general_info dev.Devices.general_info.Devices.id)
      | _ -> (
          Cuda.cuda_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols dev.Devices.general_info dev.Devices.general_info.Devices.id))
   | Devices.OpenCLInfo _ -> 
     (match Vector.vector vecA with
      | Vector.CustomArray custom ->  (OpenCL.opencl_custom_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols dev.Devices.general_info (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ())))
      | _ -> (OpenCL.opencl_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols dev.Devices.general_info (dev.Devices.general_info.Devices.id - (Devices.cuda_devices ()))))
  )

let cpu_matrix_copy (vecA: ('a,'b) Vector.vector) ldA start_rowA start_colA (vecB: ('a,'b) Vector.vector) ldB start_rowB start_colB rows cols =
  ()

let rec matrix_copy (vecA: ('a,'b) Vector.vector) ldA start_rowA start_colA 
    (vecB: ('a,'b) Vector.vector) ldB start_rowB start_colB rows cols =
  if (start_rowA*ldA+start_colA) + (rows*cols) - 1 > Vector.length vecA then
    raise (Invalid_argument 
             (Printf.sprintf "index out of bounds %d > %d"
                ((start_rowA*ldA+start_colA) + rows*cols - 1) (Vector.length vecA)) )
  else 
  if (start_rowB*ldB+start_colB) + (rows*cols) - 1 > Vector.length vecB then
    raise (Invalid_argument 
             (Printf.sprintf "index out of bounds %d > %d"
                ((start_rowB*ldA+start_colB) + rows*cols - 1) (Vector.length vecB)) )
  else 
    match Vector.dev vecA, Vector.dev vecB with
    | Vector.Dev d1, Vector.Dev d2 ->
      if d1 = d2 then
        gpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols d1
      else
        (Devices.flush d2 ();
         to_device vecB d1;
         Devices.flush d1 ();
         gpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols d1 )
    |  Vector.Dev d1 , Vector.Transferring d2 ->
      if d1 = d2 then
        (Devices.flush d1 ();
         gpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols d1)
      else
        (Devices.flush d2 ();
         to_device vecB d1;
         Devices.flush d1 ();
         gpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols d1)
    | Vector.Dev d1, Vector.No_dev ->
      (to_device vecB d1;
       Devices.flush d1 ();
       gpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols d1) 
    | Vector.Transferring d1 ,  _ ->
      Devices.flush d1 ();
      matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols;
    | Vector.No_dev, Vector.No_dev ->
      cpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols;
    | Vector.No_dev, Vector.Dev d2 
    | Vector.No_dev, Vector.Transferring d2 ->
      to_cpu vecB ();
      Devices.flush d2 ();
      cpu_matrix_copy vecA ldA start_rowA start_colA vecB ldB start_rowB start_colB rows cols

