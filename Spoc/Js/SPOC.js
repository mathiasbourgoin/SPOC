//Cuda functions
var noCuda = 1


//Provides: caml_sys_system_command
function caml_sys_system_command() {
    console.log("caml_sys_system_command");
    return 0;
}

//Provides: spoc_cuInit
 function spoc_cuInit() {
    console.log(" spoc_cuInit");
    return 0;
}
//Provides: spoc_cuda_compile
function spoc_cuda_compile() {
    console.log(" spoc_cuda_compile");
    return 0;
}

//Provides: spoc_cuda_debug_compile
function spoc_cuda_debug_compile() {
    console.log(" spoc_cuda_debug_compile");
    return 0;
}

//Provides: spoc_getCudaDevice
function spoc_getCudaDevice(i) {
    console.log("spoc_getCudaDevice");
    return 0;
}

//Provides: spoc_getCudaDevicesCount
function spoc_getCudaDevicesCount() {
    console.log(" spoc_getCudaDevicesCount");
    return 0;
}

///////////////

//Provides:  custom_getsizeofbool
function custom_getsizeofbool() {
    return 0;
}


var noCL = 0;

//Provides: spoc_clInit
function spoc_clInit() {
    if (window.webcl == undefined) {
      alert("Unfortunately your system does not support WebCL. " +
            "Make sure that you have both the OpenCL driver " +
            "and the WebCL browser extension installed.");
	noCL = 1;
    }
    else 
    {
        alert("CONGRATULATIONS! Your system supports WebCL");
	console.log ("INIT OPENCL");
	noCL = 0;
    }

    return 0;
}




//Provides: spoc_getOpenCLDevice

function spoc_getOpenCLDevice(relative_i, absolute_i) {
    console.log(" spoc_getOpenCLDevice");

 var infos = [ [ "DEVICE_ADDRESS_BITS", WebCL.DEVICE_ADDRESS_BITS ],
      [ "DEVICE_AVAILABLE", WebCL.DEVICE_AVAILABLE ],
      [ "DEVICE_COMPILER_AVAILABLE", WebCL.DEVICE_COMPILER_AVAILABLE ],
      [ "DEVICE_ENDIAN_LITTLE", WebCL.DEVICE_ENDIAN_LITTLE ],
      [ "DEVICE_ERROR_CORRECTION_SUPPORT", WebCL.DEVICE_ERROR_CORRECTION_SUPPORT ],
      [ "DEVICE_EXECUTION_CAPABILITIES", WebCL.DEVICE_EXECUTION_CAPABILITIES ],
      [ "DEVICE_EXTENSIONS", WebCL.DEVICE_EXTENSIONS ],
      [ "DEVICE_GLOBAL_MEM_CACHE_SIZE", WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE ],
      [ "DEVICE_GLOBAL_MEM_CACHE_TYPE", WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE ],
      [ "DEVICE_GLOBAL_MEM_CACHELINE_SIZE", WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE ],
      [ "DEVICE_GLOBAL_MEM_SIZE", WebCL.DEVICE_GLOBAL_MEM_SIZE ],
      [ "DEVICE_HALF_FP_CONFIG", WebCL.DEVICE_HALF_FP_CONFIG ],
      [ "DEVICE_IMAGE_SUPPORT", WebCL.DEVICE_IMAGE_SUPPORT ],
      [ "DEVICE_IMAGE2D_MAX_HEIGHT", WebCL.DEVICE_IMAGE2D_MAX_HEIGHT ],
      [ "DEVICE_IMAGE2D_MAX_WIDTH", WebCL.DEVICE_IMAGE2D_MAX_WIDTH ],
      [ "DEVICE_IMAGE3D_MAX_DEPTH", WebCL.DEVICE_IMAGE3D_MAX_DEPTH ],
      [ "DEVICE_IMAGE3D_MAX_HEIGHT", WebCL.DEVICE_IMAGE3D_MAX_HEIGHT ],
      [ "DEVICE_IMAGE3D_MAX_WIDTH", WebCL.DEVICE_IMAGE3D_MAX_WIDTH ],
      [ "DEVICE_LOCAL_MEM_SIZE", WebCL.DEVICE_LOCAL_MEM_SIZE ],
      [ "DEVICE_LOCAL_MEM_TYPE", WebCL.DEVICE_LOCAL_MEM_TYPE ],
      [ "DEVICE_MAX_CLOCK_FREQUENCY", WebCL.DEVICE_MAX_CLOCK_FREQUENCY ],
      [ "DEVICE_MAX_COMPUTE_UNITS", WebCL.DEVICE_MAX_COMPUTE_UNITS ],
      [ "DEVICE_MAX_CONSTANT_ARGS", WebCL.DEVICE_MAX_CONSTANT_ARGS ],
      [ "DEVICE_MAX_CONSTANT_BUFFER_SIZE", WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE ],
      [ "DEVICE_MAX_MEM_ALLOC_SIZE", WebCL.DEVICE_MAX_MEM_ALLOC_SIZE ],
      [ "DEVICE_MAX_PARAMETER_SIZE", WebCL.DEVICE_MAX_PARAMETER_SIZE ],
      [ "DEVICE_MAX_READ_IMAGE_ARGS", WebCL.DEVICE_MAX_READ_IMAGE_ARGS ],
      [ "DEVICE_MAX_SAMPLERS", WebCL.DEVICE_MAX_SAMPLERS ],
      [ "DEVICE_MAX_WORK_GROUP_SIZE", WebCL.DEVICE_MAX_WORK_GROUP_SIZE ],
      [ "DEVICE_MAX_WORK_ITEM_DIMENSIONS", WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS ],
      [ "DEVICE_MAX_WORK_ITEM_SIZES", WebCL.DEVICE_MAX_WORK_ITEM_SIZES ],
      [ "DEVICE_MAX_WRITE_IMAGE_ARGS", WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS ],
      [ "DEVICE_MEM_BASE_ADDR_ALIGN", WebCL.DEVICE_MEM_BASE_ADDR_ALIGN ],
      [ "DEVICE_NAME", WebCL.DEVICE_NAME ],
      [ "DEVICE_PLATFORM", WebCL.DEVICE_PLATFORM ],
      [ "DEVICE_PREFERRED_VECTOR_WIDTH_CHAR", WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR ],
      [ "DEVICE_PREFERRED_VECTOR_WIDTH_SHORT", WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT ],
      [ "DEVICE_PREFERRED_VECTOR_WIDTH_INT", WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT ],
      [ "DEVICE_PREFERRED_VECTOR_WIDTH_LONG", WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG ],
      [ "DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT", WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT ],
      [ "DEVICE_PROFILE", WebCL.DEVICE_PROFILE ],
      [ "DEVICE_PROFILING_TIMER_RESOLUTION", WebCL.DEVICE_PROFILING_TIMER_RESOLUTION ],
      [ "DEVICE_QUEUE_PROPERTIES", WebCL.DEVICE_QUEUE_PROPERTIES ],
      [ "DEVICE_SINGLE_FP_CONFIG", WebCL.DEVICE_SINGLE_FP_CONFIG ],
      [ "DEVICE_TYPE", WebCL.DEVICE_TYPE ],
      [ "DEVICE_VENDOR", WebCL.DEVICE_VENDOR ],
      [ "DEVICE_VENDOR_ID", WebCL.DEVICE_VENDOR_ID ],
      [ "DEVICE_VERSION", WebCL.DEVICE_VERSION ],
      [ "DRIVER_VERSION", WebCL.DRIVER_VERSION ] ];


    var total_num_devices = 0;

    var general_info = [0];
    var opencl_info = [1];

    var specific_info = new Array (48);
    specific_info[0] = 0;

    var platform_info = [0];

    var platforms = webcl.getPlatforms ();

     for (var z in platforms) {
	var plat = platforms[z];
	var devices = plat.getDevices ();
	total_num_devices += devices.length;
    }


    var current = 0;

    platforms = webcl.getPlatforms ();
    for (var i in platforms) {
	console.log("here "+i);
	var plat = platforms[i];
	var devices = plat.getDevices ();
	var num_devices = devices.length;
	console.log("there "+current+" "+num_devices+" "+relative_i);
	if ( (current + num_devices) >= relative_i) {
	    for (var d in devices){
		// looking at current device
		var dev = devices[d];
		if (current == relative_i ){
		    console.log("current ----------"+current);
		    //general info
		    general_info[1] = caml_new_string(dev.getInfo(WebCL.DEVICE_NAME));
		    console.log (general_info[1]);
		    general_info[2] = dev.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);
		    general_info[3] = dev.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);
		    general_info[4] = dev.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);
		    general_info[5] = dev.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);
		    general_info[6] = dev.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);
		    general_info[7] = dev.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);
		    general_info[8] = absolute_i;

		    var context = new Array(3); //cl_contex + 2 queues
		    context[0] = webcl.createContext (dev);
		    context[1] = context[0].createCommandQueue();
		    context[2] = context[0].createCommandQueue();
		    general_info[9] = context;


		    //platform info
		    platform_info[1] = caml_new_string(plat.getInfo(WebCL.PLATFORM_PROFILE));
		    platform_info[2] = caml_new_string(plat.getInfo(WebCL.PLATFORM_VERSION));
		    platform_info[3] = caml_new_string(plat.getInfo(WebCL.PLATFORM_NAME));
		    platform_info[4] = caml_new_string(plat.getInfo(WebCL.PLATFORM_VENDOR));
		    platform_info[5] = caml_new_string(plat.getInfo(WebCL.PLATFORM_EXTENSIONS));
		    platform_info[6] = num_devices;

		    //specific info
		    var infoType = dev.getInfo(WebCL.DEVICE_TYPE);
		    var type = 0;
		    if (infoType & WebCL.DEVICE_TYPE_CPU) 
			specific_info[2] = 0;
		    if (infoType & WebCL.DEVICE_TYPE_GPU) 
			specific_info[2] = 1;
		    if (infoType & WebCL.DEVICE_TYPE_ACCELERATOR) 
			specific_info[2] = 2;
		    if (infoType & WebCL.DEVICE_TYPE_DEFAULT) 
			specific_info[2] = 3;
		    specific_info[3] = caml_new_string(dev.getInfo(WebCL.DEVICE_PROFILE));
		    specific_info[4] = caml_new_string(dev.getInfo(WebCL.DEVICE_VERSION));
		    specific_info[5] = caml_new_string(dev.getInfo(WebCL.DEVICE_VENDOR));
		    var ext = dev.getInfo(WebCL.DEVICE_EXTENSIONS);
		    specific_info[6] = caml_new_string(ext);
		    specific_info[7] = dev.getInfo(WebCL.DEVICE_VENDOR_ID);
		    specific_info[8] = dev.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);
		    specific_info[9] = dev.getInfo(WebCL.DEVICE_ADDRESS_BITS);
		    specific_info[10] = dev.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);
		    specific_info[11] = dev.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);
		    specific_info[12] = dev.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);
		    specific_info[13] = dev.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);
		    specific_info[14] = dev.getInfo(WebCL.DEVICE_MAX_SAMPLERS);
		    specific_info[15] = dev.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);
		    //specific_info[16] = dev.getInfo(WebCL.DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
		    specific_info[17] = dev.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
		    specific_info[18] = dev.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);
		    specific_info[19] = dev.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);
		    specific_info[20] = dev.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);
		    specific_info[21] = dev.getInfo(WebCL.DEVICE_AVAILABLE);
		    specific_info[22] = dev.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);
		    specific_info[23] = dev.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);
		    specific_info[24] = dev.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);
		    specific_info[25] = dev.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);
		    specific_info[26] = dev.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);
//		    specific_info[27] = dev.getInfo(WebCL.DEVICE_DOUBLE_FP_CONFIG);
		    specific_info[28] = dev.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);
		    specific_info[29] = dev.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);
		    //specific_info[30] = dev.getInfo(WebCL.DEVICE_HALF_FP_CONFIG);
		    specific_info[31] = dev.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);
		    specific_info[32] = dev.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);
		    specific_info[33] = dev.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);
		    specific_info[34] = dev.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);
		    specific_info[35] = dev.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);
		    specific_info[36] = dev.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);
		    specific_info[37] = dev.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);
		    specific_info[38] = [0]
		    var dim_sizes = dev.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);
		    specific_info[38][1] = dim_sizes[0];
		    specific_info[38][2] = dim_sizes[1];
		    specific_info[38][3] = dim_sizes[2];

		    specific_info[39] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
		    specific_info[40] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
		    specific_info[41] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);
		    specific_info[42] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
		    specific_info[43] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
//		    specific_info[44] = dev.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE );
		    specific_info[45] = dev.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);
		    specific_info[46] = caml_new_string(dev.getInfo(WebCL.DRIVER_VERSION));
		    current++;

		    break;
		}
		else 
		{
		    current++;
		}
	    }
	}
	else 
	{
	    current += num_devices;
	}
    }
    var dev = [0];
    specific_info[1] = platform_info;
    opencl_info[1] = specific_info;
    dev[1] = general_info;
    dev[2] = opencl_info;

    return dev;
    
}



//Provides: spoc_getOpenCLDevicesCount
function spoc_getOpenCLDevicesCount() {
    console.log(" spoc_getOpenCLDevicesCount");
    var nb_devices = 0;
    var platforms = webcl.getPlatforms ();
    for (var i in platforms) {
	var plat = platforms[i];
	var devices = plat.getDevices ();
	nb_devices += devices.length;
    }
    return nb_devices;
}

//Provides: spoc_opencl_compile
function spoc_opencl_compile() {
    console.log(" spoc_opencl_compile");
    return 0;
}

//Provides: spoc_opencl_is_available
function spoc_opencl_is_available() {
    console.log(" spoc_opencl_is_available");
    return (!noCL);
}

