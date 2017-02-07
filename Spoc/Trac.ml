let jsonFileName = "profilingInfo.json"

let startTime = ref 0.0

(*let fileOutput = Unix.out_channel_of_descr (Unix.openfile jsonFileName [Unix.O_WRONLY; Unix.O_CREAT; Unix.O_TRUNC] 0o660)*)

let fileOutput = open_out_gen [Open_creat; Open_wronly; Open_trunc] 0o777 jsonFileName

let eventId = ref 0

let nextEvent () = eventId := !eventId + 1; !eventId

let getTime () = int_of_float ((Unix.gettimeofday() *. 1000000.0) -. !startTime)

let setStartTime () = startTime := Unix.gettimeofday() *. 1000000.0

let openOutput () = Printf.fprintf fileOutput "[\n"

let closeOutput () = Printf.fprintf fileOutput "{}]"; close_out fileOutput

let printEvent desc =
	let id = nextEvent () in
	let time = getTime () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"event\",\n
	\"desc\":\"%s\",\n
	\"id\":\"%i\",\n
	\"time\":\"%i\"\n
	},\n%!" desc id time

let beginEvent desc =
	let id = nextEvent () in
	let time = getTime () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"%s\",\n
	\"etat\":\"start\",\n
	\"id\":\"%i\",\n
	\"startTime\":\"%i\"\n
	},\n%!" desc id time;
	id

let endEvent desc id =
	let time = getTime () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"%s\",\n
	\"etat\":\"end\",\n
	\"id\":\"%i\",\n
	\"endTime\":\"%i\"\n" desc id time;
	Printf.fprintf fileOutput "},\n%!"

let c_event_gpu_free desc size vect_id type_gpu dev_id =
	let time = getTime () in
	let id = nextEvent () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"freeGPU\",\n
	\"desc\":\"%s\",\n
	\"time\":\"%i\",\n
	\"id\":\"%i\",\n
	\"vectorId\":\"%i\",\n
	\"vectorSize\":\"%i\",\n
	\"gpuType\":\"%s\",\n
	\"deviceId\":\"%i\"\n
	},\n%!" desc time id vect_id size type_gpu dev_id

let c_event_gpu_alloc desc size vect_id type_gpu dev_id =
	let time = getTime () in
	let id = nextEvent () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"allocGPU\",\n
	\"desc\":\"%s\",\n
	\"time\":\"%i\",\n
	\"id\":\"%i\",\n
	\"vectorId\":\"%i\",\n
	\"vectorSize\":\"%i\",\n
	\"gpuType\":\"%s\",\n
	\"deviceId\":\"%i\"\n
	},\n%!" desc time id vect_id size type_gpu dev_id

let c_event_start_transfert desc size vect_id type_gpu dev_id is_sub =
	let time = getTime () in
	let id = nextEvent () in
	Printf.fprintf fileOutput "{\n
		\"type\":\"transfert\",\n
		\"desc\":\"%s\",\n
		\"state\":\"start\",\n
		\"time\":\"%i\",\n
		\"id\":\"%i\",\n
		\"vectorId\":\"%i\",\n
		\"vectorSize\":\"%i\",\n
		\"gpuType\":\"%s\",\n
		\"deviceId\":\"%i\",\n
		\"isSub\":\"%B\"\n
		},\n%!" desc time id vect_id size type_gpu dev_id is_sub;
		id

let c_event_start_part_transfert desc part_size total_size part_id vect_id type_gpu id_device is_sub =
	let time = getTime () in
	let id = nextEvent () in
	Printf.fprintf fileOutput "{\n
		\"type\":\"part_transfert\",\n
		\"desc\":\"%s\",\n
		\"state\":\"start\",\n
		\"time\":\"%i\",\n
		\"id\":\"%i\",\n
		\"partId\":\"%i\",\n
		\"vectorId\":\"%i\",\n
		\"partSize\":\"%i\",\n
		\"vectorSize\":\"%i\",\n
		\"gpuType\":\"%s\",\n
		\"deviceId\":\"%i\",\n
		\"isSub\":\"%B\"\n
		},\n%!" desc time id part_id vect_id part_size total_size type_gpu id_device is_sub;
		id

let c_event_end_transfert vect_id duration id =
	let time = getTime () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"transfert\",\n
	\"id\":\"%i\",\n
	\"state\":\"end\",\n
	\"time\":\"%i\",\n
	\"vectorId\":\"%i\",\n
	\"duration\":\"%i\"\n
	},\n%!" id time vect_id (int_of_float duration)

let c_event_end_part_transfert part_id duration id =
	let time = getTime () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"part_transfert\",\n
	\"id\":\"%i\",\n
	\"state\":\"end\",\n
	\"time\":\"%i\",\n
	\"partId\":\"%i\",\n
	\"duration\":\"%i\"\n
	},\n%!" id time part_id (int_of_float duration)

let c_event_gpu_compile desc duration dev_id =
	let time = getTime() in
	let id = nextEvent () in
	Printf.fprintf fileOutput "{\n
	\"type\":\"compile\",\n
	\"desc\":\"%s\",\n
	\"time\":\"%i\",\n
	\"id\":\"%i\",\n
	\"duration\":\"%ld\",\n
	\"deviceId\":\"%i\"
},\n%!" desc time id duration dev_id
   

let c_event_gpu_exec_start desc dev_id =
		let time = getTime() in
		let id = nextEvent () in
		Printf.fprintf fileOutput "{\n
		\"type\":\"execution\",\n
		\"desc\":\"%s\",\n
		\"state\":\"start\",\n
		\"time\":\"%i\",\n
		\"id\":\"%i\",\n
		\"deviceId\":\"%i\"
		},\n%!" desc time id dev_id;
		id

let c_event_gpu_exec_end id dur =
		let time = getTime() in
		Printf.fprintf fileOutput "{\n
		\"type\":\"execution\",\n
		\"state\":\"end\",\n
		\"id\":\"%i\",\n
		\"time\":\"%i\",\n
		\"duration\":\"%ld\"\n
		},\n%!" id time dur

let _ = Callback.register "start_of_transfer" c_event_start_transfert;
  Callback.register "end_of_transfer" c_event_end_transfert;
  Callback.register "start_of_transfer_part" c_event_start_part_transfert;
  Callback.register "end_of_transfer_part" c_event_end_part_transfert;
  Callback.register "gpu_free" c_event_gpu_free;
  Callback.register "gpu_alloc" c_event_gpu_alloc;
  Callback.register "gpu_compile" c_event_gpu_compile;
  Callback.register "start_of_exec" c_event_gpu_exec_start;
  Callback.register "end_of_exec" c_event_gpu_exec_end;
