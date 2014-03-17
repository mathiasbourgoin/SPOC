//Provides: spoc_opencl_flush
function spoc_opencl_flush(gi, queue_id) {
    console.log("spoc_opencl_flush");
    var queue = gi[9][queue_id+1];
    queue.flush();
    gi[9][queue_id+1] = queue;
    return 0;
}

//Provides: spoc_opencl_load_param_vec
function spoc_opencl_load_param_vec(off, ker, idx, A, gi) {
    console.log("spoc_opencl_load_param_vec");
    var d_A = A[2];
    ker.setArg (off[1], d_A);
    off[1] = off[1]+1;
    return 0;
}

//Provides: spoc_opencl_load_param_int
function spoc_opencl_load_param_int(off, ker, val, gi) {
    console.log("spoc_opencl_load_param_int");
    ker.setArg (off[1], new Uint32Array([val]));
    off[1] = off[1]+1;
    return 0;
}

//Provides: spoc_opencl_launch_grid
function spoc_opencl_launch_grid(kern, grid, block, gi, queue_id) {
    console.log("spoc_opencl_launch_grid");
    var gridX = grid[1];
    var gridY = grid[2];
    var gridZ = grid[3];
    
    var blockX = block[1];
    var blockY = block[2];
    var blockZ = block[3];

    var global_dimension = new Array(3);
    global_dimension[0] = gridX*blockX;
    global_dimension[1] = gridY*blockY;
    global_dimension[2] = gridZ*blockZ;


    var work_size = new Array (3);
    work_size[0] = blockX;
    work_size[1] = blockY;
    work_size[2] = blockZ;
    var ctx = gi[9];

    var queue = ctx[queue_id+1];

    if ((blockX == 1) && (blockY == 1) && (blockZ == 1))
	queue.enqueueNDRangeKernel(kern, work_size.length, null, global_dimension);
    else
	
	queue.enqueueNDRangeKernel(kern, work_size.length, null, global_dimension, 
				   work_size);

    return 0;
}


//Provides: spoc_debug_opencl_compile
function spoc_debug_opencl_compile(src, fname, gi) {
    console.log(" spoc_debug_opencl_compile");
    console.log(src.bytes);
    var spoc_ctx = gi[9];
    var ctx = spoc_ctx[0];
    var program = ctx.createProgram(src.bytes);
    var devs = program.getInfo(WebCL.PROGRAM_DEVICES);
    program.build(devs);
    var kernel = program.createKernel(fname.bytes);
    
    spoc_ctx[0] = ctx;
    gi[9] = spoc_ctx;
    return kernel;
}
