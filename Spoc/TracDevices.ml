open Devices

(*module TDevices : TracDev (Ds:Devss) = struct*)

let emmitDim3 dim =
	Printf.fprintf Trac.fileOutput "{ \"type\":\"dim3\", \"x\":%i, \"y\":%i, \"z\":%i }\n" dim.x dim.y dim.z

let emmitGeneralInfo genInf =
	let ecc = string_of_bool genInf.eccEnabled in 
	Printf.fprintf Trac.fileOutput "{\n
	\"type\":\"generalInfo\",\n
	\"name\":\"%s\",\n
	\"totalGlobalMem\":%i,\n
	\"localMemSize\":%i,\n
	\"clockRate\":%i,\n
	\"totalConstMem\":%i,\n
	\"multiProcessorCount\":%i,\n
	\"eccEnabled\":\"%s\",\n
	\"id\":%i\n
	}\n" genInf.name genInf.totalGlobalMem genInf.localMemSize genInf.clockRate 
	genInf.totalConstMem genInf.multiProcessorCount ecc genInf.id

let emmitDevice dev =
	let devType = begin match dev.specific_info with
	| CudaInfo inf -> "Cuda"
	| OpenCLInfo inf -> "OpenCL"
	end in
	Printf.fprintf Trac.fileOutput "{\n
	\"type\":\"device\",
	\"generalInfo:\" ";
	emmitGeneralInfo dev.general_info;
	Printf.fprintf Trac.fileOutput ",
	\"specificInfo\":\"%s\"\n
	}\n" devType

let emmitDeviceList devList =
	let nb = List.length devList in 
	Printf.fprintf Trac.fileOutput "{
	\"type\":\"deviceList\"
	\"size\":%i
	\"elem\":[" nb;
	List.iteri (fun i dev -> emmitDevice dev; if(i != nb) then begin Printf.fprintf Trac.fileOutput ",\n" end) devList;
	Printf.fprintf Trac.fileOutput "]}\n"

(*end;;

module rec TDevicesImlp = TDevices(Devs)
and DevsImpl = Devices(TDevices);;*)