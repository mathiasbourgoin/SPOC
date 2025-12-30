(******************************************************************************
 * CUDA NVRTC - Runtime Compilation Bindings
 *
 * Ctypes bindings to NVIDIA Runtime Compilation library.
 * Used for JIT compilation of CUDA C source to PTX.
 ******************************************************************************)

open Ctypes
open Foreign

(** {1 Types} *)

(** NVRTC program handle *)
type nvrtc_program

let nvrtc_program : nvrtc_program structure typ = structure "nvrtcProgram_st"

let nvrtc_program_ptr : nvrtc_program structure ptr typ = ptr nvrtc_program

(** NVRTC result codes *)
type nvrtc_result =
  | NVRTC_SUCCESS
  | NVRTC_ERROR_OUT_OF_MEMORY
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE
  | NVRTC_ERROR_INVALID_INPUT
  | NVRTC_ERROR_INVALID_PROGRAM
  | NVRTC_ERROR_INVALID_OPTION
  | NVRTC_ERROR_COMPILATION
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | NVRTC_ERROR_INTERNAL_ERROR
  | NVRTC_ERROR_UNKNOWN of int

let nvrtc_result_of_int = function
  | 0 -> NVRTC_SUCCESS
  | 1 -> NVRTC_ERROR_OUT_OF_MEMORY
  | 2 -> NVRTC_ERROR_PROGRAM_CREATION_FAILURE
  | 3 -> NVRTC_ERROR_INVALID_INPUT
  | 4 -> NVRTC_ERROR_INVALID_PROGRAM
  | 5 -> NVRTC_ERROR_INVALID_OPTION
  | 6 -> NVRTC_ERROR_COMPILATION
  | 7 -> NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
  | 8 -> NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
  | 9 -> NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
  | 10 -> NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
  | 11 -> NVRTC_ERROR_INTERNAL_ERROR
  | n -> NVRTC_ERROR_UNKNOWN n

let int_of_nvrtc_result = function
  | NVRTC_SUCCESS -> 0
  | NVRTC_ERROR_OUT_OF_MEMORY -> 1
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE -> 2
  | NVRTC_ERROR_INVALID_INPUT -> 3
  | NVRTC_ERROR_INVALID_PROGRAM -> 4
  | NVRTC_ERROR_INVALID_OPTION -> 5
  | NVRTC_ERROR_COMPILATION -> 6
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE -> 7
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION -> 8
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION -> 9
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID -> 10
  | NVRTC_ERROR_INTERNAL_ERROR -> 11
  | NVRTC_ERROR_UNKNOWN n -> n

let nvrtc_result : nvrtc_result typ =
  view ~read:nvrtc_result_of_int ~write:int_of_nvrtc_result int

let string_of_nvrtc_result = function
  | NVRTC_SUCCESS -> "NVRTC_SUCCESS"
  | NVRTC_ERROR_OUT_OF_MEMORY -> "NVRTC_ERROR_OUT_OF_MEMORY"
  | NVRTC_ERROR_PROGRAM_CREATION_FAILURE ->
      "NVRTC_ERROR_PROGRAM_CREATION_FAILURE"
  | NVRTC_ERROR_INVALID_INPUT -> "NVRTC_ERROR_INVALID_INPUT"
  | NVRTC_ERROR_INVALID_PROGRAM -> "NVRTC_ERROR_INVALID_PROGRAM"
  | NVRTC_ERROR_INVALID_OPTION -> "NVRTC_ERROR_INVALID_OPTION"
  | NVRTC_ERROR_COMPILATION -> "NVRTC_ERROR_COMPILATION"
  | NVRTC_ERROR_BUILTIN_OPERATION_FAILURE ->
      "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE"
  | NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION ->
      "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION"
  | NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION ->
      "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION"
  | NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID ->
      "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID"
  | NVRTC_ERROR_INTERNAL_ERROR -> "NVRTC_ERROR_INTERNAL_ERROR"
  | NVRTC_ERROR_UNKNOWN n -> Printf.sprintf "NVRTC_ERROR_UNKNOWN(%d)" n

(** {1 Library Loading} *)

let nvrtc_lib : Dl.library option ref = ref None

let get_nvrtc_lib () =
  match !nvrtc_lib with
  | Some lib -> lib
  | None ->
      let lib =
        try Dl.dlopen ~filename:"libnvrtc.so.12" ~flags:[Dl.RTLD_LAZY]
        with _ -> (
          try Dl.dlopen ~filename:"libnvrtc.so.11" ~flags:[Dl.RTLD_LAZY]
          with _ -> (
            try Dl.dlopen ~filename:"libnvrtc.so" ~flags:[Dl.RTLD_LAZY]
            with _ -> (
              try Dl.dlopen ~filename:"libnvrtc.dylib" ~flags:[Dl.RTLD_LAZY]
              with _ -> (
                try
                  Dl.dlopen ~filename:"nvrtc64_120_0.dll" ~flags:[Dl.RTLD_LAZY]
                with _ -> failwith "NVRTC library not found"))))
      in
      nvrtc_lib := Some lib ;
      lib

let foreign_nvrtc name typ = foreign ~from:(get_nvrtc_lib ()) name typ

(** {1 Bindings} *)

(** Get NVRTC version *)
let nvrtcVersion =
  foreign_nvrtc "nvrtcVersion" (ptr int @-> ptr int @-> returning nvrtc_result)

(** Create a program from source *)
let nvrtcCreateProgram =
  foreign_nvrtc
    "nvrtcCreateProgram"
    (ptr nvrtc_program_ptr
   (* program *)
   @-> string
    (* source *)
    @-> string_opt
    (* name (optional) *)
    @-> int
    (* numHeaders *)
    @-> ptr string_opt
    (* headers *)
    @-> ptr string_opt
    @->
    (* includeNames *)
    returning nvrtc_result)

(** Destroy a program *)
let nvrtcDestroyProgram =
  foreign_nvrtc
    "nvrtcDestroyProgram"
    (ptr nvrtc_program_ptr @-> returning nvrtc_result)

(** Compile a program *)
let nvrtcCompileProgram =
  foreign_nvrtc
    "nvrtcCompileProgram"
    (nvrtc_program_ptr
   (* program *)
   @-> int
    (* numOptions *)
    @-> ptr string
    @->
    (* options *)
    returning nvrtc_result)

(** Get PTX size *)
let nvrtcGetPTXSize =
  foreign_nvrtc
    "nvrtcGetPTXSize"
    (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result)

(** Get PTX code *)
let nvrtcGetPTX =
  foreign_nvrtc
    "nvrtcGetPTX"
    (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result)

(** Get CUBIN size (for newer architectures) *)
let nvrtcGetCUBINSize =
  try
    foreign_nvrtc
      "nvrtcGetCUBINSize"
      (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result)
  with _ -> fun _ _ -> NVRTC_ERROR_INVALID_PROGRAM

(** Get CUBIN code *)
let nvrtcGetCUBIN =
  try
    foreign_nvrtc
      "nvrtcGetCUBIN"
      (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result)
  with _ -> fun _ _ -> NVRTC_ERROR_INVALID_PROGRAM

(** Get program log size *)
let nvrtcGetProgramLogSize =
  foreign_nvrtc
    "nvrtcGetProgramLogSize"
    (nvrtc_program_ptr @-> ptr size_t @-> returning nvrtc_result)

(** Get program log *)
let nvrtcGetProgramLog =
  foreign_nvrtc
    "nvrtcGetProgramLog"
    (nvrtc_program_ptr @-> ptr char @-> returning nvrtc_result)

(** Add name expression for lowered name lookup *)
let nvrtcAddNameExpression =
  foreign_nvrtc
    "nvrtcAddNameExpression"
    (nvrtc_program_ptr @-> string @-> returning nvrtc_result)

(** Get lowered name *)
let nvrtcGetLoweredName =
  foreign_nvrtc
    "nvrtcGetLoweredName"
    (nvrtc_program_ptr @-> string @-> ptr string @-> returning nvrtc_result)

(** {1 High-Level Helpers} *)

(** Exception for NVRTC errors *)
exception Nvrtc_error of nvrtc_result * string

(** Check result and raise if error *)
let check ctx result =
  match result with
  | NVRTC_SUCCESS -> ()
  | err -> raise (Nvrtc_error (err, ctx))

(** Compile CUDA source to PTX.
    @param source CUDA C source code
    @param name Optional program name
    @param arch Target architecture (e.g., "compute_75")
    @return PTX code as string *)
let compile_to_ptx ?(name = "kernel") ?(arch = "compute_70") (source : string) :
    string =
  (* Create program *)
  let prog = allocate nvrtc_program_ptr (from_voidp nvrtc_program null) in
  check
    "nvrtcCreateProgram"
    (nvrtcCreateProgram
       prog
       source
       (Some name)
       0
       (from_voidp string_opt null)
       (from_voidp string_opt null)) ;

  let prog_handle = !@prog in

  (* Set up options *)
  let arch_opt = Printf.sprintf "--gpu-architecture=%s" arch in
  let opts = CArray.of_list string [arch_opt; "-default-device"] in

  (* Compile *)
  let compile_result = nvrtcCompileProgram prog_handle 2 (CArray.start opts) in

  (* Get log regardless of result *)
  let log =
    let log_size = allocate size_t Unsigned.Size_t.zero in
    if nvrtcGetProgramLogSize prog_handle log_size = NVRTC_SUCCESS then
      let size = Unsigned.Size_t.to_int !@log_size in
      if size > 1 then begin
        let log_buf = allocate_n char ~count:size in
        if nvrtcGetProgramLog prog_handle log_buf = NVRTC_SUCCESS then
          Some (string_from_ptr log_buf ~length:(size - 1))
        else None
      end
      else None
    else None
  in

  (* Check compilation result *)
  (match compile_result with
  | NVRTC_SUCCESS -> ()
  | NVRTC_ERROR_COMPILATION ->
      let msg =
        match log with
        | Some l -> Printf.sprintf "NVRTC compilation failed:\n%s" l
        | None -> "NVRTC compilation failed (no log available)"
      in
      let _ = nvrtcDestroyProgram prog in
      failwith msg
  | err ->
      let _ = nvrtcDestroyProgram prog in
      raise (Nvrtc_error (err, "nvrtcCompileProgram"))) ;

  (* Get PTX *)
  let ptx_size = allocate size_t Unsigned.Size_t.zero in
  check "nvrtcGetPTXSize" (nvrtcGetPTXSize prog_handle ptx_size) ;

  let size = Unsigned.Size_t.to_int !@ptx_size in
  let ptx_buf = allocate_n char ~count:size in
  check "nvrtcGetPTX" (nvrtcGetPTX prog_handle ptx_buf) ;

  (* Cleanup *)
  let _ = nvrtcDestroyProgram prog in

  string_from_ptr ptx_buf ~length:(size - 1)

(** Get NVRTC version as (major, minor) *)
let get_version () : int * int =
  let major = allocate int 0 in
  let minor = allocate int 0 in
  check "nvrtcVersion" (nvrtcVersion major minor) ;
  (!@major, !@minor)

(** Check if NVRTC is available *)
let is_available () : bool =
  try
    let _ = get_nvrtc_lib () in
    true
  with _ -> false
