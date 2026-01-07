(******************************************************************************
 * CUDA Error Types - Structured Error Handling for CUDA Backend
 *
 * Provides structured error types for all CUDA backend operations including:
 * - Code generation errors (IR to CUDA translation)
 * - Runtime errors (device operations, compilation)
 * - Plugin errors (unsupported operations)
 ******************************************************************************)

(** Error types for CUDA code generation *)
type codegen_error =
  | Unknown_intrinsic of { name : string }
      (** Intrinsic function not recognized *)
  | Invalid_arg_count of { intrinsic : string; expected : int; got : int }
      (** Wrong number of arguments to intrinsic *)
  | Unsupported_construct of { construct : string; reason : string }
      (** IR construct not supported by CUDA *)
  | Type_error of { expr : string; expected : string; got : string }
      (** Type mismatch in expression *)
  | Invalid_memory_space of { decl : string; space : string }
      (** Invalid memory space for declaration *)

(** Error types for CUDA runtime operations *)
type runtime_error =
  | No_device_selected of { operation : string }
      (** Operation requires a device but none is set *)
  | Device_not_found of { device_id : int; max_devices : int }
      (** Device ID out of range *)
  | Compilation_failed of { source : string; log : string }
      (** PTX/CUDA compilation failed *)
  | Module_load_failed of { ptx_size : int; reason : string }
      (** Failed to load compiled module *)

(** Error types for CUDA plugin operations *)
type plugin_error =
  | Unsupported_source_lang of { lang : string; backend : string }
      (** Source language not supported by backend *)
  | Backend_unavailable of { reason : string }
      (** CUDA backend not available (missing drivers, etc.) *)
  | Library_not_found of { library : string; paths : string list }
      (** Required CUDA library not found *)

(** Union type for all CUDA errors *)
type cuda_error =
  | Codegen of codegen_error
  | Runtime of runtime_error
  | Plugin of plugin_error

(** Exception wrapper for CUDA errors *)
exception Cuda_error of cuda_error

(** {1 Error Construction Helpers} *)

let unknown_intrinsic name = Codegen (Unknown_intrinsic { name })

let invalid_arg_count intrinsic expected got =
  Codegen (Invalid_arg_count { intrinsic; expected; got })

let unsupported_construct construct reason =
  Codegen (Unsupported_construct { construct; reason })

let type_error expr expected got = Codegen (Type_error { expr; expected; got })

let invalid_memory_space decl space =
  Codegen (Invalid_memory_space { decl; space })

let no_device_selected operation = Runtime (No_device_selected { operation })

let device_not_found device_id max_devices =
  Runtime (Device_not_found { device_id; max_devices })

let compilation_failed source log = Runtime (Compilation_failed { source; log })

let module_load_failed ptx_size reason =
  Runtime (Module_load_failed { ptx_size; reason })

let unsupported_source_lang lang backend =
  Plugin (Unsupported_source_lang { lang; backend })

let backend_unavailable reason = Plugin (Backend_unavailable { reason })

let library_not_found library paths = Plugin (Library_not_found { library; paths })

(** {1 Error Conversion and Display} *)

(** Convert error to human-readable string *)
let to_string = function
  | Codegen err -> (
      match err with
      | Unknown_intrinsic { name } ->
          Printf.sprintf "Unknown CUDA intrinsic: %s" name
      | Invalid_arg_count { intrinsic; expected; got } ->
          Printf.sprintf
            "Intrinsic '%s' expects %d argument%s but got %d"
            intrinsic
            expected
            (if expected = 1 then "" else "s")
            got
      | Unsupported_construct { construct; reason } ->
          Printf.sprintf "Unsupported construct '%s': %s" construct reason
      | Type_error { expr; expected; got } ->
          Printf.sprintf
            "Type error in expression '%s': expected %s but got %s"
            expr
            expected
            got
      | Invalid_memory_space { decl; space } ->
          Printf.sprintf "Invalid memory space '%s' for declaration: %s" space decl)
  | Runtime err -> (
      match err with
      | No_device_selected { operation } ->
          Printf.sprintf
            "Operation '%s' requires a CUDA device but none is selected"
            operation
      | Device_not_found { device_id; max_devices } ->
          Printf.sprintf
            "Device ID %d not found (available: 0-%d)"
            device_id
            (max_devices - 1)
      | Compilation_failed { source; log } ->
          let preview =
            if String.length source > 100 then
              String.sub source 0 100 ^ "..."
            else source
          in
          Printf.sprintf
            "CUDA compilation failed for:\n%s\n\nCompiler log:\n%s"
            preview
            log
      | Module_load_failed { ptx_size; reason } ->
          Printf.sprintf
            "Failed to load PTX module (%d bytes): %s"
            ptx_size
            reason)
  | Plugin err -> (
      match err with
      | Unsupported_source_lang { lang; backend } ->
          Printf.sprintf
            "%s backend does not support %s source language"
            backend
            lang
      | Backend_unavailable { reason } ->
          Printf.sprintf "CUDA backend unavailable: %s" reason
      | Library_not_found { library; paths } ->
          Printf.sprintf
            "CUDA library '%s' not found in: %s"
            library
            (String.concat ", " paths))

(** Raise CUDA error as exception *)
let raise_error err = raise (Cuda_error err)

(** Print error to stderr *)
let print_error err = Printf.eprintf "[CUDA ERROR] %s\n%!" (to_string err)

(** Execute function with default fallback on error *)
let with_default ~default f =
  try f () with Cuda_error _ -> default

(** Convert error to Result type *)
let to_result f =
  try Ok (f ()) with Cuda_error err -> Error err

(** Map Result error to string *)
let result_to_string = function
  | Ok v -> Ok v
  | Error err -> Error (to_string err)
