// Cuda functions

//Provides: spoc_init_cuda_device_vec
function spoc_init_cuda_device_vec() {
    console.log("spoc_init_opencl_device_vec");
    return 0;
}



// WebCL

//Provides: spoc_init_opencl_device_vec
function spoc_init_opencl_device_vec() {
    console.log("spoc_init_opencl_device_vec");
    var ret = new Array (3);
    ret[0] = 0;
    return ret;
}

function typesize (b) {
    if ((b[1] instanceof Float32Array) || (b[1].constructor.name == "Float32Array"))
	return 4;
    else
    {
	console.log ("unimplemented vector type");
	console.log(b[1].constructor.name);
	return 4;
    }
}

//Provides: spoc_opencl_alloc_vect
function spoc_opencl_alloc_vect(vector, nb_device, gi) {
    console.log("spoc_opencl_alloc_vect");
    var bigarray = vector[2];
    var dev_vec_array = vector[4];
    var dev_vec = dev_vec_array[nb_device+1];
    var size = vector[5];
    var type_size = typesize(bigarray);
    var spoc_ctx = gi[9];
    var ctx = spoc_ctx[0];
    var spoc_ctx = gi[9];
    var ctx = spoc_ctx[0];
	
    var d_A = ctx.createBuffer(WebCL.MEM_READ_WRITE, size*type_size);
    
    dev_vec[2] = d_A;
    spoc_ctx[0] = ctx;
    gi[9] = spoc_ctx;
    return 0;
}


//Provides: spoc_opencl_cpu_to_device
function spoc_opencl_cpu_to_device(vector, nb_device, gi, queue_id) {
    console.log("spoc_opencl_cpu_to_device");
    var bigarray = vector[2];
    var dev_vec_array = vector[4];
    var dev_vec = dev_vec_array[nb_device+1];
    var size = vector[5];
    var type_size = typesize(bigarray);
    var spoc_ctx = gi[9];
    var ctx=spoc_ctx[0];
    var queue=spoc_ctx[queue_id+1];
    var d_A = dev_vec[2];
    
    queue.enqueueWriteBuffer(d_A, false, 0, (size*type_size), 
			     bigarray[1]);
    
    
    spoc_ctx[queue_id+1] = queue;
    spoc_ctx[0] = ctx;
    gi[9] = spoc_ctx;
    return 0;
}

//Provides: spoc_opencl_device_to_cpu
function spoc_opencl_device_to_cpu(vector, nb_device, gi, si, 
				   queue_id) {
    console.log("spoc_opencl_device_to_cpu");
    var bigarray = vector[2];
    var dev_vec_array = vector[4];
    var dev_vec = dev_vec_array[nb_device+1];
    var size = vector[5];
    var type_size = typesize(bigarray);
    var spoc_ctx = gi[9];
    var ctx=spoc_ctx[0];
    var queue=spoc_ctx[queue_id+1];
    var d_A = dev_vec[2];
    var h_A = bigarray[1];
    
    queue.enqueueReadBuffer(d_A, false, 0, size*type_size, h_A);

    spoc_ctx[queue_id+1] = queue;
    spoc_ctx[0] = ctx;
    gi[9] = spoc_ctx;
    return 0;
}
