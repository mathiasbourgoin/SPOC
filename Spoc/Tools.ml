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
external sizeofbool : unit -> int = "custom_getsizeofbool"

external boolget : Vector.customarray -> int -> bool = "custom_boolget"

external boolset : Vector.customarray -> int -> bool -> unit = "custom_boolset"


let print s = (Printf.printf "%s\n" (string_of_int s); Pervasives.flush stdout)


let iter f vect =
  ((match Vector.dev vect with
      | Vector.No_dev -> ()
      | Vector.Dev d | Vector.Transferring d ->
        (Devices.flush d ~queue_id: 0 ();
         Devices.flush d ~queue_id: 1 ();
         Mem.to_cpu vect ();
         Devices.flush d ~queue_id: 0 ()));
   for idx = 0 to (Vector.length vect) - 1 do f (Mem.unsafe_get vect idx) done)

let iteri f vect =
  ((match Vector.dev vect with
      | Vector.No_dev -> ()
      | Vector.Dev d | Vector.Transferring d ->
        (Devices.flush d ~queue_id: 0 ();
         Devices.flush d ~queue_id: 1 ();
         Mem.to_cpu vect ();
         Devices.flush d ~queue_id: 0 ()));
   for idx = 0 to (Vector.length vect) - 1 do f (Mem.unsafe_get vect idx) idx done)

let map f kind vect =
  ((match Vector.dev vect with
      | Vector.No_dev -> ()
      | Vector.Dev d | Vector.Transferring d ->
        (Devices.flush d ~queue_id: 0 ();
         Devices.flush d ~queue_id: 1 ();
         Mem.to_cpu vect ();
         Devices.flush d ~queue_id: 0 ()));
   let newvect = Vector.create kind (Vector.length vect)
   in
   (for i = 0 to Vector.length vect - 1 do
      Mem.unsafe_set newvect i (f (Mem.unsafe_get vect i))
    done;
    newvect))

let trueCustom =
  { Vector.size = sizeofbool (); Vector.get = boolget; Vector.set = boolset; }

let falseCustom =
  { Vector.size = sizeofbool (); Vector.get = boolget; Vector.set = boolset; }

let fold_left f seed vect =
  ((match Vector.dev vect with
      | Vector.No_dev -> ()
      | Vector.Dev d | Vector.Transferring d ->
        (Devices.flush d ~queue_id: 0 ();
         Devices.flush d ~queue_id: 1 ();
         Mem.to_cpu vect ();
         Devices.flush d ~queue_id: 0 ()));
   let s = ref seed
   in
   (for i = 0 to Vector.length vect - 1 do
      s := f !s (Mem.unsafe_get vect i)
    done;
    !s))

let fold_right f vect seed =
  ((match Vector.dev vect with
      | Vector.No_dev -> ()
      | Vector.Dev d | Vector.Transferring d ->
        (Devices.flush d ~queue_id: 0 ();
         Devices.flush d ~queue_id: 1 ();
         Mem.to_cpu vect ();
         Devices.flush d ~queue_id: 0 ()));
   let s = ref seed
   in
   (for i = Vector.length vect - 1 downto 0 do
      s := f (Mem.unsafe_get vect i) !s
    done;
    !s))

let vfalse = Vector.Custom trueCustom

let vtrue = Vector.Custom falseCustom

let spoc_bool = vfalse

