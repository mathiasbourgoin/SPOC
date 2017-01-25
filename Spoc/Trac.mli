val jsonFileName : string
val startTime : float ref
val fileOutput : out_channel
val eventId : int ref
val nextEvent : unit -> int
val getTime : unit -> int
val setStartTime : unit -> unit
val openOutput : unit -> unit
val closeOutput : unit -> unit
val printEvent : string -> unit
val beginEvent : string -> int
val endEvent : string -> int -> unit
