(*
         DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004

 Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. You just DO WHAT THE FUCK YOU WANT TO.
*)
(* DeviceQuery.ml                                      *)
(* Prints info on every device compatible with Spoc    *)
(* Does not use kernels                                *)
(*                                                     *)
(* Mathias Bourgoin - 2011                             *)
let devices = Spoc.Devices.init ()
in
(Printf.printf
   "DeviceQuery\nThis application prints information about every\ndevice compatible with Spoc found on your computer.\n";
 Printf.printf "Found %d devices: \n" (Spoc.Devices.gpgpu_devices ());
 Printf.printf "  ** %d Cuda devices \n" (Spoc.Devices.cuda_devices ());
 Printf.printf "  ** %d OpenCL devices \n" (Spoc.Devices.opencl_devices ());
 print_newline ();
 Printf.printf "Devices Info:\n";
 Array.iteri
   (fun i dev ->
      (Printf.printf "  Device  : %d\n" i;
       Printf.printf "    Name : %s\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.name;
       Printf.printf "    Total Global Memory : %d\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.totalGlobalMem;
       Printf.printf "    Local Memory Size : %d\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.localMemSize;
       Printf.printf "    Clock Rate : %d\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.clockRate;
       Printf.printf "    Total Constant Memory : %d\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.totalConstMem;
       Printf.printf "    Multi Processor Count : %d\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.multiProcessorCount;
       Printf.printf "    ECC Enabled : %b\n"
         dev.Spoc.Devices.general_info.Spoc.Devices.eccEnabled;
       (match dev.Spoc.Devices.specific_info with
        | Spoc.Devices.CudaInfo _ -> Printf.printf "    Powered by Cuda\n"
        | Spoc.Devices.OpenCLInfo _ -> Printf.printf "    Powered by OpenCL\n");
       match dev.Spoc.Devices.specific_info with
       | Spoc.Devices.CudaInfo ci ->
         (Printf.printf "    Driver Version %d\n"
            (ci.Spoc.Devices.driverVersion/1000);
          Printf.printf "    Cuda %d.%d compatible\n"
            ci.Spoc.Devices.major ci.Spoc.Devices.minor;
          Printf.printf "    Regs Per Block : %d\n"
            ci.Spoc.Devices.regsPerBlock;
          Printf.printf "    Warp Size : %d\n" ci.Spoc.Devices.warpSize;
          Printf.printf "    Memory Pitch : %d\n"
            ci.Spoc.Devices.memPitch;
          Printf.printf "    Max Threads Per Block : %d\n"
            ci.Spoc.Devices.maxThreadsPerBlock;
          let thread = ci.Spoc.Devices.maxThreadsDim
          in
          (Printf.printf "    Max Threads Dim : %dx%dx%d\n"
             thread.Spoc.Devices.x thread.Spoc.Devices.y
             thread.Spoc.Devices.z;
           let grid = ci.Spoc.Devices.maxGridSize
           in
           (Printf.printf "    Max Grid Size : %dx%dx%d\n"
              grid.Spoc.Devices.x grid.Spoc.Devices.y
              grid.Spoc.Devices.z;
            Printf.printf "    Texture Alignment : %d\n"
              ci.Spoc.Devices.textureAlignment;
            Printf.printf "    Device Overlap : %b\n"
              ci.Spoc.Devices.deviceOverlap;
            Printf.printf "    Kernel Exec Timeout Enabled : %b\n"
              ci.Spoc.Devices.kernelExecTimeoutEnabled;
            Printf.printf "    Integrated : %b\n"
              ci.Spoc.Devices.integrated;
            Printf.printf "    Can Map Host Memory : %b\n"
              ci.Spoc.Devices.canMapHostMemory;
            Printf.printf "    Compute Mode : %d\n"
              ci.Spoc.Devices.computeMode;
            Printf.printf "    Concurrent Kernels : %b\n"
              ci.Spoc.Devices.concurrentKernels;
            Printf.printf "    PCI Bus ID : %d\n"
              ci.Spoc.Devices.pciBusID;
            Printf.printf "    PCI Device ID : %d\n"
              ci.Spoc.Devices.pciDeviceID;
            flush stdout)))
       | Spoc.Devices.OpenCLInfo cli ->
         let platform = cli.Spoc.Devices.platform_info
         in
         (Printf.printf "    OpenCL compatible (via Platform : %s)\n"
            platform.Spoc.Devices.platform_name;
          Printf.printf "    Platform Profile : %s\n"
            platform.Spoc.Devices.platform_profile;
          Printf.printf "    Platform Version : %s\n"
            platform.Spoc.Devices.platform_version;
          Printf.printf "    Platform Vendor : %s\n"
            platform.Spoc.Devices.platform_vendor;
          Printf.printf "    Platform Extensions : %s\n"
            platform.Spoc.Devices.platform_extensions;
          Printf.printf "    Platform Number of Devices : %d\n"
            platform.Spoc.Devices.num_devices;
          Printf.printf "    Type : ";
          (match cli.Spoc.Devices.device_type with
           | Spoc.Devices.CL_DEVICE_TYPE_CPU -> Printf.printf "CPU\n"
           | Spoc.Devices.CL_DEVICE_TYPE_GPU -> Printf.printf "GPU\n"
           | Spoc.Devices.CL_DEVICE_TYPE_ACCELERATOR ->
             Printf.printf "ACCELERATOR\n"
           | Spoc.Devices.CL_DEVICE_TYPE_DEFAULT ->
             Printf.printf "DEFAULT\n");
          Printf.printf "    Profile : %s\n" cli.Spoc.Devices.profile;
          Printf.printf "    Version : %s\n" cli.Spoc.Devices.version;
          Printf.printf "    Vendor : %s\n" cli.Spoc.Devices.vendor;
          Printf.printf "    Driver : %s\n"
            cli.Spoc.Devices.driver_version;
          Printf.printf "    Extensions : %s\n"
            cli.Spoc.Devices.extensions;
          Printf.printf "    Vendor ID : %d\n"
            cli.Spoc.Devices.vendor_id;
          Printf.printf "    Max Work Iem Dimensions : %d\n"
            cli.Spoc.Devices.max_work_item_dimensions;
          Printf.printf "    Max Work Group Size : %d\n"
            cli.Spoc.Devices.max_work_group_size;
          Printf.printf "    Max Work Item Size : %dx%dx%d\n"
            cli.Spoc.Devices.max_work_item_size.Spoc.Devices.x
            cli.Spoc.Devices.max_work_item_size.Spoc.Devices.y
            cli.Spoc.Devices.max_work_item_size.Spoc.Devices.z;
          Printf.printf "    Address Bits : %d\n"
            cli.Spoc.Devices.address_bits;
          Printf.printf "    Max Memory Alloc Size : %d\n"
            cli.Spoc.Devices.max_mem_alloc_size;
          Printf.printf "    Image Support : %b\n"
            cli.Spoc.Devices.image_support;
          Printf.printf "    Max Read Image Args : %d\n"
            cli.Spoc.Devices.max_read_image_args;
          Printf.printf "    Max Write Image Args : %d\n"
            cli.Spoc.Devices.max_write_image_args;
          Printf.printf "    Max Samplers : %d\n"
            cli.Spoc.Devices.max_samplers;
          Printf.printf "    Memory Base Addr Align : %d\n"
            cli.Spoc.Devices.mem_base_addr_align;
          Printf.printf "    Min Data Type Align Size : %d\n"
            cli.Spoc.Devices.min_data_type_align_size;
          Printf.printf "    Global Mem Cacheline Size : %d\n"
            cli.Spoc.Devices.global_mem_cacheline_size;
          Printf.printf "    Global Mem Cache Size : %d\n"
            cli.Spoc.Devices.global_mem_cache_size;
          Printf.printf "    Max Constant Args : %d\n"
            cli.Spoc.Devices.max_constant_args;
          Printf.printf "    Endian Little : %b\n"
            cli.Spoc.Devices.endian_little;
          Printf.printf "    Available : %b\n"
            cli.Spoc.Devices.available;
          Printf.printf "    Compiler Available : %b\n"
            cli.Spoc.Devices.compiler_available;
          Printf.printf "    CL Device Single FP Config : ";
          (match cli.Spoc.Devices.single_fp_config with
           | Spoc.Devices.CL_FP_DENORM -> Printf.printf "FP DENORM\n"
           | Spoc.Devices.CL_FP_INF_NAN -> Printf.printf "FP INF NAN\n"
           | Spoc.Devices.CL_FP_ROUND_TO_NEAREST ->
             Printf.printf "FP ROUND TO NEAREST\n"
           | Spoc.Devices.CL_FP_ROUND_TO_ZERO ->
             Printf.printf "FP ROUND TO ZERO\n"
           | Spoc.Devices.CL_FP_ROUND_TO_INF ->
             Printf.printf "FP ROUND TO INF\n"
           | Spoc.Devices.CL_FP_FMA -> Printf.printf "FP FMA\n"
           | _ -> failwith "error single_fp_config");
          Printf.printf "    CL Device Double FP Config : ";
          (match cli.Spoc.Devices.double_fp_config with
           | Spoc.Devices.CL_FP_DENORM -> Printf.printf "FP DENORM\n"
           | Spoc.Devices.CL_FP_INF_NAN -> Printf.printf "FP INF NAN\n"
           | Spoc.Devices.CL_FP_ROUND_TO_NEAREST ->
             Printf.printf "FP ROUND TO NEAREST\n"
           | Spoc.Devices.CL_FP_ROUND_TO_ZERO ->
             Printf.printf "FP ROUND TO ZERO\n"
           | Spoc.Devices.CL_FP_ROUND_TO_INF ->
             Printf.printf "FP ROUND TO INF\n"
           | Spoc.Devices.CL_FP_FMA -> Printf.printf "FP FMA\n"
           | Spoc.Devices.CL_FP_NONE -> Printf.printf "FP NONE\n");
          Printf.printf "    CL Device Half FP Config : ";
          (match cli.Spoc.Devices.half_fp_config with
           | Spoc.Devices.CL_FP_DENORM -> Printf.printf "FP DENORM\n"
           | Spoc.Devices.CL_FP_INF_NAN -> Printf.printf "FP INF NAN\n"
           | Spoc.Devices.CL_FP_ROUND_TO_NEAREST ->
             Printf.printf "FP ROUND TO NEAREST\n"
           | Spoc.Devices.CL_FP_ROUND_TO_ZERO ->
             Printf.printf "FP ROUND TO ZERO\n"
           | Spoc.Devices.CL_FP_ROUND_TO_INF ->
             Printf.printf "FP ROUND TO INF\n"
           | Spoc.Devices.CL_FP_FMA -> Printf.printf "FP FMA\n"
           | Spoc.Devices.CL_FP_NONE -> Printf.printf "FP NONE\n");
          Printf.printf "    CL Device Global Mem Cache Type : ";
          (match cli.Spoc.Devices.global_mem_cache_type with
           | Spoc.Devices.CL_READ_WRITE_CACHE ->
             Printf.printf "READ WRITE CACHE\n"
           | Spoc.Devices.CL_READ_ONLY_CACHE ->
             Printf.printf "READ ONLY CACHE\n"
           | Spoc.Devices.CL_NONE -> Printf.printf "NONE\n");
          Printf.printf "    CL Device Queue Properties : ";
          (match cli.Spoc.Devices.queue_properties with
           | Spoc.Devices.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ->
             Printf.printf "OUT OF ORDER EXEC MODE ENABLE\n"
           | Spoc.Devices.CL_QUEUE_PROFILING_ENABLE ->
             Printf.printf "PROFILING ENABLE\n");
          Printf.printf "    CL Local Mem Type : ";
          (match cli.Spoc.Devices.local_mem_type with
           | Spoc.Devices.CL_LOCAL -> Printf.printf "Local\n"
           | Spoc.Devices.CL_GLOBAL -> Printf.printf "Global\n");
          Printf.printf "    Image2D DIM : %dx%d\n"
            cli.Spoc.Devices.image2D_max_height
            cli.Spoc.Devices.image2D_max_width;
          Printf.printf "    Image3D DIM : %dx%dx%d\n"
            cli.Spoc.Devices.image3D_max_depth
            cli.Spoc.Devices.image3D_max_height
            cli.Spoc.Devices.image3D_max_width;
          Printf.printf "    Preferred Vector Width Char : %d\n"
            cli.Spoc.Devices.prefered_vector_width_char;
          Printf.printf "    Preferred Vector Width Short : %d\n"
            cli.Spoc.Devices.prefered_vector_width_short;
          Printf.printf "    Preferred Vector Width Int : %d\n"
            cli.Spoc.Devices.prefered_vector_width_int;
          Printf.printf "    Preferred Vector Width Long : %d\n"
            cli.Spoc.Devices.prefered_vector_width_long;
          Printf.printf "    Preferred Vector Width Float : %d\n"
            cli.Spoc.Devices.prefered_vector_width_float;
          Printf.printf "    Preferred Vector Width Double : %d\n"
            cli.Spoc.Devices.prefered_vector_width_double;
          Printf.printf "    Profiling Timer Resolution : %d\n"
            cli.Spoc.Devices.profiling_timer_resolution;
          for j = 0 to (Spoc.Devices.cuda_devices ()) - 1 do
            if
              (String.compare
                 dev.Spoc.Devices.general_info.Spoc.Devices.name
                 devices.(j).Spoc.Devices.general_info.Spoc.Devices.
                   name)
              = 0
            then
              Printf.printf "    !!Warning!! could be Device %d\n"
                (i - (Spoc.Devices.cuda_devices ()))
            else ()
          done)
      ))
   devices)
