(******************************************************************************
 * Device type - SDK layer
 *
 * Just the type definition. Runtime functions are in spoc_core.Device.
 ******************************************************************************)

(** Device representation *)
type t = {
  id : int;  (** Global device ID (0, 1, 2...) *)
  backend_id : int;  (** ID within the backend (0, 1...) *)
  name : string;  (** Human-readable device name *)
  framework : string;  (** Backend name: "CUDA", "OpenCL", "Vulkan", "Native" *)
  capabilities : Framework_sig.capabilities;
}
