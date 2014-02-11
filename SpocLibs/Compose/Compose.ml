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
open Spoc
open Kernel 

let kernel_max ((spoc_var0 : (('a,'b)  Vector.vector)),
                (spoc_var1 : ('c,'d)  Vector.vector), (spoc_var2 : int)) = 
  Kernel.exec
    [| Kernel.VFloat64 spoc_var0; Kernel.VFloat64 spoc_var1;
      Kernel.Int32 spoc_var2|]
  
class ['a,'b] class_kernel_max =
  object (self)
    inherit
      [((float, Bigarray.float64_elt)  Vector.vector *(float, Bigarray.float64_elt)  Vector.vector * int)
			, ('a,'b) Kernel.kernelArgs array] Kernel.spoc_kernel "kernels/Spoc_kernels" "spoc_max"
      
    method exec = kernel_max
      
    method args_to_list =
      fun
        ((spoc_var0 :('c,'d)  Vector.vector),
         (spoc_var1 :('e,'f)  Vector.vector), (spoc_var2 : int))
        ->
        [| Kernel.VFloat64 spoc_var0; Kernel.VFloat64 spoc_var1;
          Kernel.Int32 spoc_var2|]
      
    method list_to_args =
      function
      | [| VFloat64 spoc_var0; VFloat64 spoc_var1; Int32 spoc_var2 |] ->
          ((spoc_var0 : ('c,'d)  Vector.vector),
           (spoc_var1 : ('e,'f)  Vector.vector), 
					(spoc_var2 : int))
      | _  -> failwith "should not hskeletonen"
      
  end
  
let spoc_max = new class_kernel_max

class ['a,'b] class_dummy_kernel =
 object (self)
    inherit
      ['a, 'b] Kernel.spoc_kernel "" ""
      
    method exec = failwith "dummy kernel"
    method args_to_list = failwith "dummy kernel"
    method list_to_args = failwith "dummy kernel"
      
  end

let spoc_dummy_kernel = new class_dummy_kernel


type reduction =
	| Max

type application =
	| Map
	| Pipe
	| Reduce
	| Par
	| Filter
			

class type  ['a,'b, 'c] skeleton =
	object
	val kind : application
	val ker: ('a,'b) Kernel.spoc_kernel
  val env : 'a
	method run :
		?queue_id:int -> Devices.device -> ('f,'g) Vector.vector -> (('d,'e) Vector.vector as 'c)
	
	method kind : unit  -> application
	method ker : unit  -> ('a,'b) Kernel.spoc_kernel
 method env : unit  -> 'a

	method par_run :
  Devices.device list -> ('f,'g) Vector.vector -> (('d,'e) Vector.vector as 'c)
	end

   

class ['a,'b, 'c] map (kern) (out:'c) (env:'a)  =
	object (self : ('a, 'b, 'c) #skeleton)
	val kind = Map;
	val ker = (kern :> ('a,'b) Kernel.spoc_kernel);
  val env = env;
  method ker () = ker;
  method kind () = (kind: application)
  method env () = env
	method run : 'e 'f.
		?queue_id:int -> Devices.device -> ('e,'f) Vector.vector -> 'c = 
      fun ?queue_id:(q=0) device  -> 
        fun inp -> 
				let block = {blockX = 1; blockY = 1; blockZ = 1}
				and grid = {gridX = 1; gridY = 1; gridZ = 1}
				in
				begin
					let open Devices in(
					match device.Devices.specific_info with
						| Devices.CudaInfo cI-> 
							if Vector.length inp < 
								(cI.maxThreadsDim.x) then
				(
					grid.gridX <- 1;
					block.blockX <-(Vector.length inp)
				)
			else
				(
					block.blockX <- cI.maxThreadsDim.x;
					grid.gridX <- (Vector.length inp) /  cI.maxThreadsDim.x;
				)
		| Devices.OpenCLInfo oI -> 
			if Vector.length inp < oI.Devices.max_work_item_size.Devices.x then
				(
					grid.gridX <- 1;
					block.blockX <- Vector.length inp
				)
			else
				(
					block.blockX <- oI.Devices.max_work_item_size.Devices.x;
					grid.gridX <- (Vector.length inp) / block.blockX
				)
		)
    end;
			let env2  = (((ker:> ('a,'b) Kernel.spoc_kernel)#args_to_list (env:'a)):'b)  in
      Kernel.set_arg (env2)  0 ((Obj.magic inp):'c);
	    (ker)#run (ker#list_to_args env2)
		        (block,grid) q device ;
	    (out)
	
	method par_run  : 'e 'f.
  Devices.device list -> ('e,'f) Vector.vector -> 'c = 
      fun devices  ->
        fun inp  ->  
		let device = List.hd devices in
			self#run device inp 
	end

let get_vector karg =
	match karg with 
		| Kernel.VFloat64 v -> v
		| _ -> failwith "not implemented"

let transfer_if_vector a device queue=
	match a with
		| Kernel.VFloat64 v -> Mem.to_device v ~queue_id: queue device;
    | Kernel.VFloat32 v -> Mem.to_device v ~queue_id: queue device;
		| _ -> ()


class ['a,'b, 'c] pipe (skeleton1: ('a,'b,'g)#skeleton) (skeleton2:('i,'j,'c) #skeleton) =
		object (self : ('a,'b,'c) #skeleton)
		val kind  = (Pipe : application)
    val env = skeleton1#env()		
    val ker = (skeleton1#ker() :> ('a,'b) Kernel.spoc_kernel)
    method env () = env
    method kind () = kind
    method ker () = ker
		
    method run : 'e 'f 'g 'i 'j.    
    ?queue_id:int -> Devices.device -> ('e,'f) Vector.vector -> 'c =      
    fun ?queue_id:(q=0) device  -> 
      fun inp ->
				
      (	
        match (skeleton1 :> ('a,'b,'g) skeleton)#kind() with
							| Map -> 
								Array.iter (fun a -> transfer_if_vector a device 0) 
									((skeleton1#ker())#args_to_list 
                  (skeleton1#env())) 
							| _ -> ()
				);


       (		match (skeleton2 :> ('i,'j,'c) skeleton)#kind() with
							| Map -> 
								Array.iter (fun a -> transfer_if_vector a device 1) 
									((skeleton2#ker())#args_to_list (skeleton2#env()))  
							| _ -> ()
				);
        
        let (v: 'g) = ((skeleton1:> ('a, 'b, 'g) skeleton)#run ~queue_id:0 device inp) in 
			  Devices.flush device ~queue_id:0 ();
				let res = (((skeleton2:> ('i, 'j, 'c) skeleton)#run ~queue_id:1 device v): 'c) in
        Devices.flush device ~queue_id:1 ();
        res

    
    method par_run =
      fun devices  ->
        fun inp  ->  
		      let device = List.hd devices in
			      self#run device inp 
		end

class ['a,'b, 'c] reduce reduction (out) (env:'a) = 
	object (self : ('a,'b,'c) #skeleton)
		val kind = Reduce
		val ker = (reduction :> ('a,'b) Kernel.spoc_kernel)
		val env = env
    method kind () = kind
    method ker () = ker
    method env () = env
    method run : 'e 'f.
    ?queue_id:int -> Devices.device -> ('e,'f) Vector.vector -> 'c = 
      fun ?queue_id:(q=0) device  ->   
						fun inp ->
						(reduction :> ('a,'b) Kernel.spoc_kernel)#compile ~debug:true device;
						let block = {blockX = 1; blockY = 1; blockZ = 1}
						and grid = {gridX = 1; gridY = 1; gridZ = 1}
						in
            let env2  = (((reduction:> ('a,'b) Kernel.spoc_kernel)#args_to_list (env:'a)):'b)  in
            Kernel.set_arg (env2)  0 (Obj.magic inp);
	          (reduction)#run (reduction#list_to_args env2)
		        (block,grid) q device ;
	    (out)  
    method par_run =
      fun devices  ->
        fun inp  ->  
		      let device = List.hd devices in
			      self#run device inp 
		end
	


let run (c: ('a,'b,'c) #skeleton) = c#run
let par_run (c: ('a,'b,'c) #skeleton) = c#par_run


let pipe (skeleton1: ('a,'b,'c) #skeleton) (skeleton2: ('d,'e,'f) #skeleton) = new pipe skeleton1 skeleton2	

let map ker out env = new map ker out env

let reduce reduction k = new reduce reduction k
