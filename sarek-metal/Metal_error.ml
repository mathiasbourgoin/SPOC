(******************************************************************************
 * Metal_error - Error Handling for Metal Backend
 *
 * Uses the shared Backend_error module instantiated for Metal backend.
 ******************************************************************************)

include Spoc_framework.Backend_error.Make (struct
  let name = "Metal"
end)

(** Re-export Backend_error exception for pattern matching *)
exception Metal_error = Spoc_framework.Backend_error.Backend_error
