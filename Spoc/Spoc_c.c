/******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
*******************************************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>



#include "Spoc.h"
int noCL = 1;


value spoc_getOpenCLDevicesCount()
{
	cl_uint num_entries = 10;
	cl_platform_id platform_ids[10];
	const cl_uint max_num_devices = 80;
	int platform_id;
	cl_device_id *device_ids;

	cl_uint  num_platforms;
	cl_uint num_devices;
	cl_uint total_num_devices = 0;
	if (noCL) return(Val_int(0));

	device_ids = malloc(sizeof(cl_device_id)*max_num_devices);
	cl_int err;
	OPENCL_TRY2 ("clGetPlatformIds", clGetPlatformIDs ( num_entries, platform_ids, &num_platforms), err);
	if (CL_SUCCESS != err)
	  raise_constant(*caml_named_value("no_platform")) ;
	for(platform_id = 0; platform_id < num_platforms; platform_id++) {
		OPENCL_TRY("clGetDeviceIDs", clGetDeviceIDs( platform_ids[platform_id], CL_DEVICE_TYPE_ALL, max_num_devices, device_ids, &num_devices));
		total_num_devices += num_devices;
	}
	free(device_ids);
	return (Val_int(total_num_devices));

}

value spoc_opencl_is_available(value i)
{
	CAMLparam1(i);
	CAMLlocal4(dev, general_info, opencl_info, specific_info);
	CAMLlocal3(platform_info,  maxT, maxG);
	cl_int device_id = Int_val(i);

	cl_uint num_entries = 10;
	cl_platform_id platform_ids[10];
	const cl_uint max_num_devices = 80;
	int platform_id;
	cl_device_id *device_ids;

	cl_uint  num_platforms;

	cl_uint infoUInt;
	cl_ulong infoULong;
	cl_bool infoBool;
    cl_device_type infoType;
    cl_device_mem_cache_type infoMemType = 0;
    cl_device_fp_config singleFPConfig;
    cl_command_queue_properties commandQueueProperties;
	char infoStr[1024];
	cl_device_local_mem_type localMemType;
	size_t infoLen;
	cl_uint num_devices;
	cl_uint total_num_devices = 0;
	cl_context ctx;
	cl_command_queue queue[2];
	cl_context_properties properties[3];
	cl_device_id dev_id[1];
	spoc_cl_context *spoc_ctx;
	int opencl_error;

	device_ids = malloc(sizeof(cl_device_id)*max_num_devices);
	OPENCL_TRY ("clGetPlatformIds", clGetPlatformIDs ( num_entries, platform_ids, &num_platforms));

	for(platform_id = 0; platform_id < num_platforms; platform_id++) {
		OPENCL_TRY("clGetDeviceIDs", clGetDeviceIDs( platform_ids[platform_id], CL_DEVICE_TYPE_ALL, max_num_devices, device_ids, &num_devices));
		total_num_devices += num_devices;
	}
	if ((Int_val(i)) > total_num_devices)
		raise_constant(*caml_named_value("no_opencl_device")) ;


	for(platform_id = 0; platform_id < num_platforms; platform_id++) {
		OPENCL_TRY("clGetDeviceIDs", clGetDeviceIDs( platform_ids[platform_id], CL_DEVICE_TYPE_ALL, max_num_devices, device_ids, &num_devices));

		if (num_devices > device_id)
		{

            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_AVAILABLE , sizeof(infoBool), &infoBool, &infoLen ));
            break;
		}else device_id -=num_devices;
	}

	free(device_ids);
	CAMLreturn(Val_bool(infoBool));


}

void pfn_notify(const char *errinfo, const void *private_info, size_t cb, void *user_data)
{
   fprintf(stderr, "%s\n", errinfo);
   fflush(stderr);
}

value spoc_getOpenCLDevice(value relative_i, value absolute_i)
{
	CAMLparam2(relative_i, absolute_i);
	CAMLlocal4(dev, general_info, opencl_info, specific_info);
	CAMLlocal3(platform_info,  maxWI, maxG);
	cl_int device_id = Int_val(relative_i);

	cl_uint num_entries = 10;
	cl_platform_id platform_ids[10];
	const cl_uint max_num_devices = 80;
	int platform_id;
	cl_device_id *device_ids;
	int i;
	cl_uint  num_platforms;

	cl_uint infoUInt;
	cl_ulong infoULong;
	cl_bool infoBool;
    cl_device_type infoType;
    cl_device_mem_cache_type infoMemType = 0;
    cl_device_fp_config fPConfig;
    cl_device_exec_capabilities execCapabilities;

    cl_command_queue_properties commandQueueProperties;
	char infoStr[1024];
	cl_device_local_mem_type localMemType;
	size_t infoLen;
	size_t infoSize;
	cl_uint num_devices;
	cl_uint infoDimension;
	cl_uint total_num_devices = 0;
	cl_context ctx;
	size_t *work_sizes;
	cl_command_queue queue[2];
	cl_context_properties properties[3];
	cl_device_id dev_id[1];
	spoc_cl_context *spoc_ctx;
	int opencl_error;

	device_ids = malloc(sizeof(cl_device_id)*max_num_devices);
	OPENCL_TRY ("clGetPlatformIds", clGetPlatformIDs ( num_entries, platform_ids, &num_platforms));

	for(platform_id = 0; platform_id < num_platforms; platform_id++) {
		OPENCL_TRY("clGetDeviceIDs", clGetDeviceIDs( platform_ids[platform_id], CL_DEVICE_TYPE_ALL, max_num_devices, device_ids, &num_devices));
		total_num_devices += num_devices;
	}
	if ((Int_val(relative_i)) > total_num_devices)
		raise_constant(*caml_named_value("no_opencl_device")) ;


	general_info = caml_alloc (9, 0);
	opencl_info = caml_alloc(1, 1); //1 -> OpenCL
	specific_info = caml_alloc(47, 0);
    platform_info = caml_alloc(6, 0);

	for(platform_id = 0; platform_id < num_platforms; platform_id++) {
		OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_NAME, sizeof(infoStr), infoStr, &infoLen ));
		OPENCL_TRY("clGetDeviceIDs", clGetDeviceIDs( platform_ids[platform_id], CL_DEVICE_TYPE_ALL, max_num_devices, device_ids, &num_devices));

		if (num_devices > device_id)
		{
			//general info
			OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_NAME, sizeof(infoStr), infoStr, &infoLen ));
			Store_field(general_info, 0, copy_string(infoStr));
			OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_GLOBAL_MEM_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
			Store_field(general_info,1, Val_int(infoULong));
			OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_LOCAL_MEM_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
			Store_field(general_info,2, Val_int(infoULong));
			OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(infoUInt), &infoUInt, &infoLen ));
			Store_field(general_info,3, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
            Store_field(general_info,4, Val_int(infoULong));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(infoUInt), &infoUInt, &infoLen ));
			Store_field(general_info,5, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_ERROR_CORRECTION_SUPPORT , sizeof(infoBool), &infoBool, &infoLen ));
            Store_field(general_info,6, Val_bool(infoBool));
            Store_field(general_info,7, absolute_i);
            properties[0] = CL_CONTEXT_PLATFORM;
            properties[1] = (cl_context_properties) platform_ids[platform_id];
            properties[2] = 0;
            dev_id[0] = device_ids[device_id];
            OPENCL_CHECK_CALL1(ctx, clCreateContext(properties, 1, dev_id, pfn_notify, NULL, &opencl_error));
        	OPENCL_CHECK_CALL1(queue[0],  clCreateCommandQueue(ctx, dev_id[0], CL_QUEUE_PROFILING_ENABLE, &opencl_error));
        	OPENCL_CHECK_CALL1(queue[1],  clCreateCommandQueue(ctx, dev_id[0], CL_QUEUE_PROFILING_ENABLE, &opencl_error));
        	spoc_ctx = malloc(sizeof(spoc_cl_context));
        	spoc_ctx->ctx = ctx;
        	spoc_ctx->queue[0] = queue[0];
        	spoc_ctx->queue[1] = queue[1];
        	OPENCL_CHECK_CALL1(opencl_error, clRetainCommandQueue(queue[0]));
        	OPENCL_CHECK_CALL1(opencl_error, clRetainCommandQueue(queue[1]));

            Store_field(general_info,8, (value) spoc_ctx);


            //platform info
            OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_PROFILE, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(platform_info, 0, copy_string(infoStr));
            OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_VERSION, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(platform_info, 1, copy_string(infoStr));
            OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_NAME, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(platform_info, 2, copy_string(infoStr));
            OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_VENDOR, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(platform_info, 3, copy_string(infoStr));
            OPENCL_TRY("clGetPlatformInfo", clGetPlatformInfo ( platform_ids[platform_id], CL_PLATFORM_EXTENSIONS, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(platform_info, 4, copy_string(infoStr));
            Store_field(platform_info, 5, Val_int(num_devices));

            //specific info
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_TYPE, sizeof(infoType), &infoType, &infoLen ));
            if (infoType & CL_DEVICE_TYPE_CPU)
            	Store_field(specific_info, 1, Val_int(0));
            if (infoType & CL_DEVICE_TYPE_GPU)
            	Store_field(specific_info, 1,Val_int(1));
            if (infoType & CL_DEVICE_TYPE_ACCELERATOR)
            	Store_field(specific_info, 1,Val_int(2));
            if (infoType & CL_DEVICE_TYPE_DEFAULT)
            	Store_field(specific_info, 1,Val_int(3));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PROFILE, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(specific_info, 2, copy_string(infoStr));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_VERSION, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(specific_info, 3, copy_string(infoStr));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_VENDOR, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(specific_info, 4, copy_string(infoStr));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_EXTENSIONS, sizeof(infoStr), infoStr, &infoLen ));
            Store_field(specific_info, 5, copy_string(infoStr));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_VENDOR_ID, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 6, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(infoDimension), &infoDimension, &infoLen));
            Store_field(specific_info, 7, Val_int(infoDimension));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_ADDRESS_BITS, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 8, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_MEM_ALLOC_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
            Store_field(specific_info, 9, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE_SUPPORT , sizeof(infoBool), &infoBool, &infoLen ));
            Store_field(specific_info, 10, Val_bool(infoBool));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 11, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 12, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_SAMPLERS, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 13, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 14, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(infoUInt), &infoUInt, &infoLen));
            Store_field(specific_info, 15, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(infoUInt), &infoUInt, &infoLen));
            Store_field(specific_info, 16, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
            Store_field(specific_info, 17, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(infoUInt), &infoUInt, &infoLen ));
            Store_field(specific_info, 18, Val_int(infoUInt));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_ENDIAN_LITTLE , sizeof(infoBool), &infoBool, &infoLen ));
            Store_field(specific_info, 19, Val_bool(infoBool));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_AVAILABLE , sizeof(infoBool), &infoBool, &infoLen ));
            Store_field(specific_info, 20, Val_bool(infoBool));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_COMPILER_AVAILABLE , sizeof(infoBool), &infoBool, &infoLen ));
            Store_field(specific_info, 21, Val_bool(infoBool));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &fPConfig, &infoLen ));
            if(fPConfig & CL_FP_DENORM)
            	Store_field(specific_info, 22, Val_int(0));
            if(fPConfig & CL_FP_INF_NAN)
            	Store_field(specific_info, 22, Val_int(1));
            if(fPConfig & CL_FP_ROUND_TO_NEAREST)
            	Store_field(specific_info, 22, Val_int(2));
            if(fPConfig & CL_FP_ROUND_TO_ZERO)
            	Store_field(specific_info, 22, Val_int(3));
            if(fPConfig & CL_FP_ROUND_TO_INF)
            	Store_field(specific_info, 22, Val_int(4));
            if(fPConfig & CL_FP_FMA)
            	Store_field(specific_info, 22, Val_int(5));
           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(infoMemType), &infoMemType,  &infoLen ));
            if(infoMemType == CL_READ_WRITE_CACHE)
               	Store_field(specific_info, 23, Val_int(0));
            if(infoMemType == CL_READ_ONLY_CACHE)
            	Store_field(specific_info, 23, Val_int(1));
            if(infoMemType == CL_NONE)
            	Store_field(specific_info, 23, Val_int(2));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_QUEUE_PROPERTIES, sizeof(commandQueueProperties), &commandQueueProperties, &infoLen ));
            if(commandQueueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
            	Store_field(specific_info, 24, Val_int(0));
            if(commandQueueProperties & CL_QUEUE_PROFILING_ENABLE)
            	Store_field(specific_info, 24, Val_int(1));
            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(localMemType), &localMemType, &infoLen ));
            if(localMemType == CL_LOCAL)
            	Store_field(specific_info, 25, Val_int(0));
            if(localMemType == CL_GLOBAL)
            	Store_field(specific_info, 25, Val_int(1));

            /***************/
            if (strstr(String_val(Field(specific_info, 5)), "cl_khr_fp64")){

            OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &fPConfig, &infoLen ));
            if(fPConfig & CL_FP_DENORM)
            	Store_field(specific_info, 26, Val_int(0));
            if(fPConfig & CL_FP_INF_NAN)
            	Store_field(specific_info, 26, Val_int(1));
            if(fPConfig & CL_FP_ROUND_TO_NEAREST)
               	Store_field(specific_info, 26, Val_int(2));
            if(fPConfig & CL_FP_ROUND_TO_ZERO)
               	Store_field(specific_info, 26, Val_int(3));
            if(fPConfig & CL_FP_ROUND_TO_INF)
               	Store_field(specific_info, 26, Val_int(4));
            if(fPConfig & CL_FP_FMA)
               	Store_field(specific_info, 26, Val_int(5));
            }
            else
               	Store_field(specific_info, 26, Val_int(6));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE , sizeof(infoULong), &infoULong, &infoLen ));
           Store_field(specific_info, 27, Val_int(infoUInt));


           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_EXECUTION_CAPABILITIES , sizeof(cl_device_exec_capabilities), &execCapabilities, &infoLen ));
           if(execCapabilities == CL_EXEC_KERNEL)
        	   Store_field(specific_info, 28, Val_int(0));
           if(execCapabilities == CL_EXEC_NATIVE_KERNEL)
           	   Store_field(specific_info, 28, Val_int(1));

           if (strstr(String_val(Field(specific_info, 5)), "cl_khr_fp16")){
           OPENCL_TRY(
						"clGetDeviceInfo",
						clGetDeviceInfo(device_ids[device_id],
								CL_DEVICE_HALF_FP_CONFIG,
								sizeof(cl_device_fp_config), &fPConfig,
								&infoLen));
				if (fPConfig & CL_FP_DENORM)
					Store_field(specific_info, 29, Val_int(0));
				if (fPConfig & CL_FP_INF_NAN)
					Store_field(specific_info, 29, Val_int(1));
				if (fPConfig & CL_FP_ROUND_TO_NEAREST)
					Store_field(specific_info, 29, Val_int(2));
				if (fPConfig & CL_FP_ROUND_TO_ZERO)
					Store_field(specific_info, 29, Val_int(3));
				if (fPConfig & CL_FP_ROUND_TO_INF)
					Store_field(specific_info, 29, Val_int(4));
				if (fPConfig & CL_FP_FMA)
					Store_field(specific_info, 29, Val_int(5));
           }
           else	Store_field(specific_info, 29, Val_int(6));


           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 30, Val_int((int)infoSize));


           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE2D_MAX_HEIGHT , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 31, Val_int((int)infoSize));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE2D_MAX_WIDTH , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 32, Val_int((int)infoSize));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE3D_MAX_DEPTH , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 33, Val_int((int)infoSize));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE3D_MAX_HEIGHT , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 34, Val_int((int)infoSize));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_IMAGE3D_MAX_WIDTH , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 35, Val_int((int)infoSize));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_PARAMETER_SIZE , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 36, Val_int((int)infoSize));

           work_sizes = malloc(sizeof(size_t)*infoDimension);
           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_MAX_WORK_ITEM_SIZES , sizeof(size_t)*infoDimension, work_sizes, &infoLen ));
           if (infoDimension > 3) infoDimension = 3; //for now limited to 3d
           maxWI = caml_alloc(infoDimension, 0);
           for (i = 0; i < infoDimension; i++)
        	   Store_field(maxWI ,i, Val_int(work_sizes[i]));
           Store_field(specific_info, 37, maxWI);
	   free(work_sizes);

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 38, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 39, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 40, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 41, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 42, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE , sizeof(infoUInt), &infoUInt, &infoLen ));
           Store_field(specific_info, 43, Val_int(infoUInt));

           OPENCL_TRY("clGetDeviceInfo", clGetDeviceInfo ( device_ids[device_id], CL_DEVICE_PROFILING_TIMER_RESOLUTION , sizeof(infoSize), &infoSize, &infoLen ));
           Store_field(specific_info, 44, Val_int((int)infoSize));

           OPENCL_TRY("clGetPlatformInfo", clGetDeviceInfo ( device_ids[device_id], CL_DRIVER_VERSION, sizeof(infoStr), infoStr, &infoLen ));
           Store_field(specific_info, 45, copy_string(infoStr));

           break;

		}else device_id -=num_devices;
	}

	dev = caml_alloc(2, 0);
	Store_field(specific_info, 0, platform_info);
	Store_field(opencl_info, 0, specific_info);
	Store_field(dev, 0, general_info);
	Store_field(dev, 1, opencl_info);
	free(device_ids);
	CAMLreturn(dev);


}


CAMLprim value float32_of_float (value f){
	CAMLparam1(f);
	CAMLlocal1(v);
	float fl = (float)Double_val(f);
	v = caml_copy_double(fl);
	CAMLreturn(v);
}

CAMLprim value float_of_float32 (value f){
	CAMLparam1(f);
	CAMLlocal1(v);
	double fl = (double)Double_val(f);
	v = caml_copy_double(fl);
	CAMLreturn(v);
}

cuComplex complex_val (value c){
	cuComplex res;
	res.x = (float)(Double_val(Field(0,c)));
	res.y = (float)(Double_val(Field(1,c)));
	return res;
}


cuDoubleComplex doubleComplex_val (value c){
	cuDoubleComplex res;
	res.x = (Double_val(Field(0,c)));
	res.y = (Double_val(Field(1,c)));
	return res;
}

value copy_two_doubles(double d0, double d1){
  value res = caml_alloc_small(2 * Double_wosize, Double_array_tag);
  Store_double_field(res, 0, d0);
  Store_double_field(res, 1, d1);
  return res;
}

value copy_complex(cuComplex d){
  return copy_two_doubles((double)d.x, (double)d.y);
}

value copy_doubleComplex(cuDoubleComplex d){
  return copy_two_doubles(d.x, d.y);
}
