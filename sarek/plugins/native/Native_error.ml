(** Error handling for Native backend plugin *)

include Spoc_framework.Backend_error.Make (struct
  let name = "Native"
end)
