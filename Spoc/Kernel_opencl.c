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
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <math.h>
#include <string.h>
#include "Spoc.h"

void p (char* s){
	printf ("%s", s);
	fflush (stdout);
}

CAMLprim value spoc_opencl_compile(value moduleSrc, value function_name, value gi){
	CAMLparam3(moduleSrc, function_name, gi);
	cl_program hProgram;
	cl_device_id device_id;
	cl_kernel kernel;
	char* functionN;
	char* cl_source;
	cl_int ret_val;
	 char *paramValue;
	size_t paramValueSize, param_value_size_ret;

	OPENCL_GET_CONTEXT;

	functionN = String_val(function_name);
	cl_source = String_val(moduleSrc);

	OPENCL_CHECK_CALL1(hProgram, clCreateProgramWithSource(ctx, 1, (const char**)&cl_source, 0, &opencl_error));
	OPENCL_TRY("clGetContextInfo", clGetContextInfo(ctx, CL_CONTEXT_DEVICES, (size_t)sizeof(cl_device_id), &device_id, NULL)) ;
	OPENCL_CHECK_CALL1(ret_val, clBuildProgram(hProgram, 1, &device_id, 0, NULL, NULL));

	paramValueSize = 1024 * 1024;

	OPENCL_CHECK_CALL1(kernel, clCreateKernel(hProgram, functionN, &opencl_error));
	OPENCL_RESTORE_CONTEXT;
	
	CAMLreturn((value) kernel);;

}


CAMLprim value spoc_debug_opencl_compile(value moduleSrc, value function_name, value gi){
	CAMLparam3(moduleSrc, function_name, gi);
	cl_program hProgram;
	cl_device_id device_id;
	cl_kernel kernel;
	char* functionN;
	char* cl_source;
	cl_int ret_val;
	size_t paramValueSize,  param_value_size_ret;
    char *paramValue;


	OPENCL_GET_CONTEXT;

	functionN = String_val(function_name);
	cl_source = String_val(moduleSrc);

	OPENCL_CHECK_CALL1(hProgram, clCreateProgramWithSource(ctx, 1, (const char**)&cl_source, 0, &opencl_error));
	OPENCL_TRY("clGetContextInfo", clGetContextInfo(ctx, CL_CONTEXT_DEVICES, (size_t)sizeof(cl_device_id), &device_id, NULL)) ;
	OPENCL_CHECK_CALL1(ret_val, clBuildProgram(hProgram, 1, &device_id, 0, NULL, NULL));

	paramValueSize = 1024 * 1024;

	paramValue = (char*)calloc(paramValueSize, sizeof(char));
	ret_val = clGetProgramBuildInfo( hProgram,
					 device_id,
					 CL_PROGRAM_BUILD_LOG,
					 paramValueSize,
	                           paramValue,
					 &param_value_size_ret);
	
	fprintf(stdout, " %s" , paramValue);
	free(paramValue);
	OPENCL_CHECK_CALL1(kernel, clCreateKernel(hProgram, functionN, &opencl_error));
	OPENCL_RESTORE_CONTEXT;
	CAMLreturn((value) kernel);;

}


CAMLprim value spoc_opencl_create_dummy_kernel(){
	CAMLparam0();
	cl_kernel kernel = NULL;
	CAMLreturn((value) kernel);
}

CAMLprim value spoc_opencl_load_param_vec(value off, value ker, int idx, value A, value gi){
	CAMLparam4(off, ker, A, gi);
	cl_kernel kernel;
	cl_mem d_A;
	int offset;
	offset = Int_val(Field(off, 0));
		d_A = Cl_mem_val(Field(A, 1));
	OPENCL_GET_CONTEXT;

	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(cl_mem), (void*)&d_A));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_load_param_local_vec(value off, value ker, int idx, value A, value gi){
	CAMLparam4(off, ker, A, gi);
	cl_kernel kernel;
	cl_mem d_A;
	int offset;

	offset = Int_val(Field(off, 0));
		d_A = Cl_mem_val(Field(A, 1));

	OPENCL_GET_CONTEXT;

	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(float)*4*8*8, NULL));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_load_param_int(value off, value ker, value val, value gi){
	CAMLparam4(off, ker, val, gi);
	cl_kernel kernel;
	int i;
	int offset;
	offset = Int_val(Field(off, 0));

	OPENCL_GET_CONTEXT;
	i = Int_val(val);
	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(int), (void*)&i));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_load_param_int64(value off, value ker, value val, value gi){
	CAMLparam4(off, ker, val, gi);
	cl_kernel kernel;
	long long i;
	int offset;
	offset = Int_val(Field(off, 0));

	OPENCL_GET_CONTEXT;

	i = (long long)Int_val(val);
	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(long long), (void*)&i));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_load_param_float(value off, value ker, value val, value gi){
	CAMLparam4(off, ker, val, gi);
	cl_kernel kernel;
	float f;
	int offset;
	offset = Int_val(Field(off, 0));

	OPENCL_GET_CONTEXT;

	f = (float)(Double_val(val));
	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(float), (void*)&f));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_load_param_float64(value off, value ker, value val, value gi){
	CAMLparam4(off, ker, val, gi);
	cl_kernel kernel;
	double f;
	int offset;
	offset = Int_val(Field(off, 0));

	OPENCL_GET_CONTEXT;

	f = (Double_val(val));
	kernel = (cl_kernel) ker;
	OPENCL_CHECK_CALL1(opencl_error, clSetKernelArg(kernel, offset, sizeof(double), (void*)&f));
	offset+=1;
	OPENCL_RESTORE_CONTEXT;
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_opencl_launch_grid(value ker, value grid, value block, value gi, value queue_id){
	CAMLparam5(ker, grid, block, gi, queue_id);
	cl_kernel kernel;
	unsigned int gridX, gridY, gridZ, blockX, blockY, blockZ;
	unsigned int dimension;
	size_t work_size [3];
	size_t global_dimension [3];
	cl_command_queue q;
	gridX = Int_val(Field(grid,0));
	gridY = Int_val(Field(grid,1));
	gridZ = Int_val(Field(grid,2));

	blockX = Int_val(Field(block,0));
	blockY = Int_val(Field(block,1));
	blockZ = Int_val(Field(block,2));
	OPENCL_GET_CONTEXT;

	kernel = (cl_kernel) ker;
	
	global_dimension[0] = (size_t)gridX*blockX;
	global_dimension[1] = (size_t)gridY*blockY;
	global_dimension[2] =(size_t)gridZ*blockZ;
	work_size[0] = (size_t)blockX;
	work_size[1] = (size_t)blockY;
	work_size[2] = (size_t)blockZ;
	
	q = queue[Int_val(queue_id)];
	OPENCL_CHECK_CALL1(opencl_error, clRetainCommandQueue(queue[Int_val(queue_id)]));
	OPENCL_CHECK_CALL1(opencl_error, clEnqueueNDRangeKernel
			   (q, kernel, 3, NULL, global_dimension,
			    ((blockX == 1) && (blockY == 1) && (blockZ == 1)) ? (size_t *) NULL : work_size,
			    0, NULL, NULL));
	OPENCL_CHECK_CALL1(opencl_error, clReleaseCommandQueue(queue[Int_val(queue_id)]));
	
	OPENCL_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}


CAMLprim value spoc_opencl_flush(value gi, value queue_id){
	CAMLparam2(gi, queue_id);
	cl_command_queue q;
	OPENCL_GET_CONTEXT;
   	q = queue[Int_val(queue_id)];
	OPENCL_CHECK_CALL1(opencl_error, clFinish(q));
	OPENCL_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}


#ifdef __cplusplus
}
#endif
