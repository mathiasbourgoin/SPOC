(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
(* SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com> *)
(******************************************************************************)

(******************************************************************************
 * Host_async - the [Framework_sig.ASYNC] building blocks for backends whose
 * device memory IS host memory (Native, Interpreter).
 *
 * On such a backend a "device" buffer is a host allocation, so a ranged copy
 * is a memmove, a stream is synchronous (every enqueued operation has already
 * completed when the call returns), an event is always complete, and pinned
 * memory is ordinary host memory: page-locking has no meaning when no DMA
 * engine reads it.
 ******************************************************************************)

(** Copy [byte_size] bytes from address [src] to address [dst] (memmove
    semantics: Bigarray blit tolerates overlap). Both ranges must be valid; the
    caller keeps them alive across the call. *)
let blit ~(src : nativeint) ~(dst : nativeint) ~(byte_size : int) : unit =
  if byte_size > 0 then begin
    let view addr =
      Ctypes.bigarray_of_ptr
        Ctypes.array1
        byte_size
        Bigarray.char
        (Ctypes.from_voidp Ctypes.char (Ctypes.ptr_of_raw_address addr))
    in
    Bigarray.Array1.blit (view src) (view dst)
  end

(** Check a byte range against a buffer of [total] bytes; raise
    [Invalid_argument] naming [what] when it does not fit. *)
let check_range ~(what : string) ~(total : int) ~(off : int) ~(byte_size : int)
    : unit =
  if off < 0 || byte_size < 0 || off + byte_size > total then
    invalid_arg
      (Printf.sprintf
         "%s: byte range [%d, %d) outside buffer of %d bytes"
         what
         off
         (off + byte_size)
         total)

(* Host allocations handed out as "pinned" memory, keyed by address. The table
   is what keeps each Bigarray alive until [free_pinned]. *)
let pinned :
    ( nativeint,
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    )
    Hashtbl.t =
  Hashtbl.create 8

let pinned_lock = Mutex.create ()

(** Ordinary host memory standing in for page-locked memory. *)
let alloc_pinned ~(byte_size : int) : nativeint =
  if byte_size <= 0 then
    invalid_arg "Host_async.alloc_pinned: byte_size must be positive" ;
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout byte_size in
  let addr =
    Ctypes.raw_address_of_ptr
      (Ctypes.to_voidp (Ctypes.bigarray_start Ctypes.array1 ba))
  in
  Mutex.protect pinned_lock (fun () -> Hashtbl.replace pinned addr ba) ;
  addr

let free_pinned (addr : nativeint) : unit =
  Mutex.protect pinned_lock (fun () ->
      if not (Hashtbl.mem pinned addr) then
        invalid_arg "Host_async.free_pinned: not a pinned allocation" ;
      Hashtbl.remove pinned addr)
