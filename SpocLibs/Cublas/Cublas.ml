(******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011) * *
   Mathias.Bourgoin@gmail.com * * This software is a computer program
   whose purpose is to allow * GPU programming with the OCaml language.
 * * This software is governed by the CeCILL-B license under French law
   and * abiding by the rules of distribution of free software.  You can
   use, * modify and/ or redistribute the software under the terms of the
   CeCILL-B * license as circulated by CEA, CNRS and INRIA at the
   following URL * "http://www.cecill.info".  * * As a counterpart to the
   access to the source code and rights to copy, * modify and
   redistribute granted by the license, users are provided only * with a
   limited warranty and the software's author, the holder of the *
   economic rights, and the successive licensors have only limited *
   liability.  * * In this respect, the user's attention is drawn to the
   risks associated * with loading, using, modifying and/or developing or
   reproducing the * software by the user in light of its specific status
   of free software, * that may mean that it is complicated to
   manipulate, and that also * therefore means that it is reserved for
   developers and experienced * professionals having in-depth computer
   knowledge. Users are therefore * encouraged to load and test the
   software's suitability as regards their * requirements in conditions
   enabling the security of their systems and/or * data to be ensured
   and, more generally, to use and operate it in the * same conditions as
   regards security.  * * The fact that you are presently reading this
   means that you have had * knowledge of the CeCILL-B license and that
   you accept its terms.
 *******************************************************************************)

open Spoc

exception NOT_INITIALIZED

exception ALLOC_FAILED

exception INVALID_VALUE

exception ARCH_MISMATCH

exception MAPPING_ERROR

exception EXECUTION_FAILED

exception INTERNAL_ERROR

exception UNKNOWN_ERROR

let _ =
  Callback.register_exception "cublas_status_not_initialised" NOT_INITIALIZED ;
  Callback.register_exception "cublas_status_alloc_failed" ALLOC_FAILED ;
  Callback.register_exception "cublas_status_invalid_value" INVALID_VALUE ;
  Callback.register_exception "cublas_status_arch_mismatch" ARCH_MISMATCH ;
  Callback.register_exception "cublas_status_mapping_error" MAPPING_ERROR ;
  Callback.register_exception "cublas_status_execution_failed" EXECUTION_FAILED ;
  Callback.register_exception "cublas_status_internal_error" INTERNAL_ERROR ;
  Callback.register_exception "cublas_error_unknown" UNKNOWN_ERROR

external init : unit -> unit = "spoc_cublasInit"

external shutdown : unit -> unit = "spoc_cublasShutdown"

external getError : unit -> unit = "spoc_cublasGetError"

type vfloat32 = (float, Bigarray.float32_elt) Spoc.Vector.vector

type vchar = (char, Bigarray.int8_unsigned_elt) Spoc.Vector.vector

type vfloat64 = (float, Bigarray.float64_elt) Spoc.Vector.vector

type vcomplex32 = (Complex.t, Bigarray.complex32_elt) Spoc.Vector.vector

(* Blas1 *)
external cublasIsamax : int -> vfloat32 -> int -> Devices.device -> int
  = "spoc_cublasIsamax_b" "spoc_cublasIsamax_n"

external cublasIsamin : int -> vfloat32 -> int -> Devices.device -> int
  = "spoc_cublasIsamin_b" "spoc_cublasIsamin_n"

external cublasSasum : int -> vfloat32 -> int -> Devices.device -> float
  = "spoc_cublasSasum_b" "spoc_cublasSasum_n"

external cublasSaxpy :
  int -> float -> vfloat32 -> int -> vfloat32 -> int -> Devices.device -> unit
  = "spoc_cublasSaxpy_b" "spoc_cublasSaxpy_n"

external cublasScopy :
  int -> vfloat32 -> int -> vfloat32 -> int -> Devices.device -> unit
  = "spoc_cublasScopy_b" "spoc_cublasScopy_n"

external cublasSdot :
  int -> vfloat32 -> int -> vfloat32 -> int -> Devices.device -> float
  = "spoc_cublasSdot_b" "spoc_cublasSdot_n"

external cublasSnrm2 : int -> vfloat32 -> int -> Devices.device -> float
  = "spoc_cublasSnrm2_b" "spoc_cublasSnrm2_n"

external cublasSrot :
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  float ->
  float ->
  Devices.device ->
  unit = "spoc_cublasSrot_b" "spoc_cublasSrot_n"

external cublasSrotg : vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> unit
  = "spoc_cublasSrotg_b" "spoc_cublasSrotg_n"

external cublasSrotm :
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  vfloat32 ->
  Devices.device ->
  unit = "spoc_cublasSrotm_b" "spoc_cublasSrotm_n"

external cublasSrotmg :
  vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> vfloat32 -> unit
  = "spoc_cublasSrotmg_b" "spoc_cublasSrotmg_n"

external cublasSscal : int -> float -> vfloat32 -> int -> Devices.device -> unit
  = "spoc_cublasSscal_b" "spoc_cublasSscal_n"

external cublasSswap :
  int -> vfloat32 -> int -> vfloat32 -> int -> Devices.device -> unit
  = "spoc_cublasSswap_b" "spoc_cublasSswap_n"

external cublasCaxpy :
  int ->
  Complex.t ->
  vcomplex32 ->
  int ->
  vcomplex32 ->
  int ->
  Devices.device ->
  unit = "spoc_cublasCaxpy_b" "spoc_cublasCaxpy_n"

external cublasScasum : int -> vcomplex32 -> int -> Devices.device -> float
  = "spoc_cublasScasum_b" "spoc_cublasScasum_n"

let rec check_auto_cpu vecs =
  if !Mem.auto then
    for i = 0 to Array.length vecs - 1 do
      match Vector.dev vecs.(i) with
      | Vector.No_dev -> ()
      | Vector.Transferring d ->
          Devices.flush d ~queue_id:0 () ;
          Devices.flush d ~queue_id:1 () ;
          check_auto_cpu vecs
      | Vector.Dev d ->
          Devices.flush d () ;
          Mem.to_cpu vecs.(i) () ;
          Devices.flush d ()
    done

let rec check_auto vecs dev =
  if !Mem.auto then
    for i = 0 to Array.length vecs - 1 do
      match Vector.dev vecs.(i) with
      | Vector.No_dev -> Mem.to_device vecs.(i) dev
      | Vector.Transferring d ->
          Devices.flush d ~queue_id:0 () ;
          Devices.flush d ~queue_id:1 () ;
          check_auto vecs dev
      | Vector.Dev d ->
          if Vector.device vecs.(i) != d.Devices.general_info.Devices.id then
            Devices.flush d () ;
          Mem.to_device vecs.(i) dev
    done ;
  Devices.flush dev ()

(** Blas1 *)

(** Single-Precision Blas1 functions *)

let cublasIsamax n x incx dev =
  check_auto [|x|] dev ;
  cublasIsamax n x incx dev

let cublasIsamin n x incx dev =
  check_auto [|x|] dev ;
  cublasIsamin n x incx dev

let cublasSasum n x incx dev =
  check_auto [|x|] dev ;
  cublasSasum n x incx dev

let cublasSaxpy n alpha x incx y incy dev =
  check_auto [|x; y|] dev ;
  cublasSaxpy n alpha x incx y incy dev

let cublasScopy n x incx y incy dev =
  check_auto [|x; y|] dev ;
  cublasScopy n x incx y incy dev

let cublasSdot n x incx y incy dev =
  check_auto [|x; y|] dev ;
  cublasSdot n x incx y incy dev

let cublasSnrm2 n x incx dev =
  check_auto [|x|] dev ;
  cublasSnrm2 n x incx dev

let cublasSrot n x incx y incy sc ss dev =
  check_auto [|x; y|] dev ;
  cublasSrot n x incx y incy sc ss dev

let cublasSrotg host_sa host_sb host_sc host_ss dev =
  let vecs = [|host_sa; host_sb; host_sc; host_ss|] in
  if !Mem.auto then begin
    for i = 0 to Array.length vecs - 1 do
      if Vector.device vecs.(i) != -1 then Mem.to_cpu vecs.(i) ()
    done
  end ;
  cublasSrotg host_sa host_sb host_sc host_ss

let cublasSrotm n x incx y incy sparam dev =
  check_auto [|x; y; sparam|] dev ;
  cublasSrotm n x incx y incy sparam dev

let cublasSrotmg host_psd1 host_psd2 host_psx1 host_psy1 host_sparam dev =
  let vecs = [|host_psd1; host_psd2; host_psx1; host_psy1; host_sparam|] in
  if !Mem.auto then begin
    for i = 0 to Array.length vecs - 1 do
      if Vector.device vecs.(i) != -1 then Mem.to_cpu vecs.(i) ()
    done
  end ;
  cublasSrotmg host_psd1 host_psd2 host_psx1 host_psy1 host_sparam

let cublasSscal n alpha x incx dev =
  check_auto [|x|] dev ;
  cublasSscal n alpha x incx dev

let cublasSswap n x incx y incy dev =
  check_auto [|x; y|] dev ;
  cublasSswap n x incx y incy dev

(** Single-Precision Complex Blas1 functions *)

let cublasCaxpy n alpha x incx y incy dev =
  check_auto [|x; y|] dev ;
  cublasCaxpy n alpha x incx y incy dev

let cublasScasum n x incx dev =
  check_auto [|x|] dev ;
  cublasScasum n x incx dev

(*Blas3*)

(** Blas3 *)

(** Single-Precision Blas3 functions *)

external cublasSgemm :
  char ->
  char ->
  int ->
  int ->
  int ->
  float ->
  vfloat32 ->
  int ->
  vfloat32 ->
  int ->
  float ->
  vfloat32 ->
  int ->
  Devices.device ->
  unit = "spoc_cublasSgemm_b" "spoc_cublasSgemm_n"

let cublasSgemm transa transb m n k alpha a lda b ldb beta c ldc dev =
  check_auto [|a; b; c|] dev ;
  cublasSgemm transa transb m n k alpha a lda b ldb beta c ldc dev

(** Double-Precision Blas3 functions *)

external cublasDgemm :
  char ->
  char ->
  int ->
  int ->
  int ->
  float ->
  vfloat64 ->
  int ->
  vfloat64 ->
  int ->
  float ->
  vfloat64 ->
  int ->
  Devices.device ->
  unit = "spoc_cublasDgemm_b" "spoc_cublasDgemm_n"

let cublasDgemm transa transb m n k alpha a lda b ldb beta c ldc dev =
  check_auto [|a; b; c|] dev ;
  cublasDgemm transa transb m n k alpha a lda b ldb beta c ldc dev

let run dev f = f dev

external cublasSetMatrix :
  int ->
  int ->
  ('a, 'b) Spoc.Vector.vector ->
  int ->
  ('a, 'b) Spoc.Vector.vector ->
  int ->
  Spoc.Devices.device ->
  unit = "spoc_cublasSetMatrix_b" "spoc_cublasSetMatrix_n"

let setMatrix rows cols a lda b ldb dev =
  if (rows - 1) * (cols - 1) > Vector.length b then
    raise (Invalid_argument "index out of bound") ;
  check_auto_cpu [|a|] ;
  check_auto [|b|] dev ;
  cublasSetMatrix rows cols a lda b ldb dev
