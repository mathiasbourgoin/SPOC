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
/**************** KERNEL ******************/

int ae_load_file_to_memory(const char *filename, char **result)
{
	int size = 0;
	FILE *f = fopen(filename, "rb");
	if (f == NULL)
	{
		*result = NULL;
		return -1; // -1 means file opening fail
	}
	fseek(f, 0, SEEK_END);
	size = ftell(f);
	fseek(f, 0, SEEK_SET);
	*result = (char *)malloc(size+1);
	if (size != fread(*result, sizeof(char), size, f))
	{
		free(*result);
		return -2; // -2 means file reading fail
	}
	fclose(f);
	(*result)[size] = 0;
	return size;
}


CAMLprim value spoc_cuda_compile(value moduleSrc, value function_name, value gi){
	CAMLparam3(moduleSrc, function_name, gi);
	CUmodule module;
	CUfunction *kernel;
	char* functionN;
	char *ptx_source;
	const unsigned int jitNumOptions = 4;

	CUjit_option jitOptions[4];
	void *jitOptVals[4];
	int jitLogBufferSize;
	char *jitLogBuffer;
	int jitRegCount = 32;

	CUDA_GET_CONTEXT;

	kernel = malloc(sizeof(CUfunction));
	functionN = String_val(function_name);
	ptx_source = String_val(moduleSrc);

	// set up size of compilation log buffer
	jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	jitLogBufferSize = 1024;
	jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

	// set up pointer to the compilation log buffer
	jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
	jitLogBuffer = malloc(sizeof(char)*jitLogBufferSize);
	jitOptVals[1] = jitLogBuffer;

	// set up pointer to set the Maximum # of registers for a particular kernel
	jitOptions[2] = CU_JIT_MAX_REGISTERS;
	jitOptVals[2] = (void *)(size_t)jitRegCount;

	// set up pointer to set the Maximum # of registers for a particular kernel
	jitOptions[3] = CU_JIT_TARGET_FROM_CUCONTEXT;
	//CU_JIT_TARGET;
//	jitOptVals[3] =  (void*)(uintptr_t)CU_TARGET_COMPUTE_11;

	CUDA_CHECK_CALL(cuModuleLoadDataEx(&module, ptx_source, jitNumOptions, jitOptions, (void **)jitOptVals));
	CUDA_CHECK_CALL(cuModuleGetFunction(kernel, module, functionN));
	free(jitLogBuffer);
	CUDA_RESTORE_CONTEXT;
	//caml_leave_blocking_section();
	CAMLreturn((value) kernel);
}


CAMLprim value spoc_cuda_debug_compile(value moduleSrc, value function_name, value gi){
	CAMLparam3(moduleSrc, function_name, gi);
	CUmodule module;
	CUfunction *kernel;
	char* functionN;
	char *ptx_source;
	const unsigned int jitNumOptions = 4;

	CUjit_option jitOptions[4];
	void *jitOptVals[4];
	int jitLogBufferSize;
	char *jitLogBuffer;
	int jitRegCount = 32;

	BLOCKING_CUDA_GET_CONTEXT;

	kernel = malloc(sizeof(CUfunction));
	functionN = String_val(function_name);
	ptx_source = String_val(moduleSrc);

	// set up size of compilation log buffer
	jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
	jitLogBufferSize = 1024;
	jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

	// set up pointer to the compilation log buffer
	jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
	jitLogBuffer = malloc(sizeof(char)*jitLogBufferSize);
	jitOptVals[1] = jitLogBuffer;

	// set up pointer to set the Maximum # of registers for a particular kernel
	jitOptions[2] = CU_JIT_MAX_REGISTERS;
	jitOptVals[2] = (void *)(size_t)jitRegCount;

	jitOptions[3] = CU_JIT_TARGET_FROM_CUCONTEXT;
	//CU_JIT_TARGET;
//	jitOptVals[3] =  (void*)(uintptr_t)CU_TARGET_COMPUTE_10;


	cuda_error = (cuModuleLoadDataEx(&module, ptx_source, jitNumOptions, jitOptions, (void **)jitOptVals));
	if (cuda_error)
		{
			printf ("%s\n", jitLogBuffer);
			fflush (stdout);
		}
	cuda_error = (cuModuleGetFunction(kernel, module, functionN));
	if (cuda_error)
	{
		printf ("%s\n", jitLogBuffer);
		fflush (stdout);
	}
	BLOCKING_CUDA_RESTORE_CONTEXT;
	free(jitLogBuffer);
	CAMLreturn((value) kernel);
}


CAMLprim value spoc_cuda_create_extra(value n){
	CAMLparam1(n);
	char *extra = malloc(sizeof(void*)*Int_val(n)*16);
	CAMLreturn ((value)extra);
}

CAMLprim value spoc_cuda_create_dummy_kernel(){
	CAMLparam0();
	CUfunction *kernel = NULL;
	CAMLreturn((value) kernel);
}

#define ALIGN_UP(offset, alignment) \
(offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)


#define ADD_TO_PARAM_BUFFER(value, alignment)\
do {\
	offset = ALIGN_UP(offset, alignment); \
	memcpy(extra + offset, &(value), sizeof(value));\
	offset += sizeof(value);\
} while (0)



CAMLprim value spoc_cuda_load_param_vec(value off, value ex, value A, value v, value device){
	CAMLparam4(off, ex, A, v);
	CAMLlocal1(bigArray);
	cu_vector* cuv;
	CUdeviceptr d_A;
	void* ptr;
	char *extra;
	int offset;
	int seek;
	int type_size;
	int tag;


	seek = Int_val(Field(v,10));
	bigArray = Field (Field(v, 1), 0);
	int custom = 0;
	GET_TYPE_SIZE;
	extra = (char*)ex;
	offset = Int_val(Field(off, 0));
	cuv = (cu_vector*)Field(A, 1);
	d_A = cuv->cu_vector;
	ptr = (void*) (size_t) d_A+seek*type_size;
	ADD_TO_PARAM_BUFFER(ptr, __alignof(d_A));
	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_custom_load_param_vec(value off, value ex, value A, value v){
	CAMLparam4(off, ex, A, v);
	CAMLlocal1(customArray);
	cu_vector* cuv;
	CUdeviceptr d_A;
	char *extra;
	int offset;
	int seek;
	int type_size;
	int tag;
	void* ptr;
	seek = Int_val(Field(v,10));
	customArray = Field (Field(v, 1), 0);
	type_size = Int_val(Field(Field(customArray, 1),1));

	extra = (char*)ex;
	offset = Int_val(Field(off, 0));

	cuv = (cu_vector*)Field(A, 1);
	d_A = cuv->cu_vector;
	ptr = (void*) (size_t) d_A + seek * type_size;
	ADD_TO_PARAM_BUFFER(ptr, __alignof(d_A));

	Store_field(off, 0, Val_int(offset));

	CAMLreturn(Val_unit);
}


CAMLprim value spoc_cuda_load_param_int(value off, value ex,value val){
	CAMLparam3(off, ex, val);
	int offset, i;// = malloc(sizeof(i));
	char*extra;
	extra = (char*)ex;
	offset = Int_val(Field(off, 0));
	i = Int_val(val);
	ADD_TO_PARAM_BUFFER(i, __alignof(i));
	Store_field(off, 0, Val_int(offset));
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_load_param_int64(value off, value ex, value val){
	CAMLparam3(off, ex, val);
	int offset;
	char *extra;
	long long i;
	extra = (char*)ex;
	offset = Int_val(Field(off, 0));
	i = (long long) Int_val(val);
	ADD_TO_PARAM_BUFFER(i, __alignof(i));
	Store_field(off, 0, Val_int(offset));
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_load_param_float(value off, value ex, value val){
	CAMLparam3(off, ex, val);
	int offset;
	char *extra;
	float f;
	offset = Int_val(Field(off, 0));
	extra = (char*)ex;
	f = (float)Double_val(val);
	ADD_TO_PARAM_BUFFER(f, __alignof(f));
	Store_field(off, 0, Val_int(offset));
	CAMLreturn(Val_unit);
}
CAMLprim value spoc_cuda_load_param_float64(value off, value ex, value val){
	CAMLparam3(off, ex, val);
	int offset;
	char *extra;
	double f;
	extra = (char*)ex;
	offset = Int_val(Field(off, 0));
	f = Double_val(val);
	ADD_TO_PARAM_BUFFER(f, __alignof(f));
	Store_field(off, 0, Val_int(offset));
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_load_param_vec_b(value off, value ex, value A, value v, value d){
	return spoc_cuda_load_param_vec(off, ex, A, v, d);

}
CAMLprim value spoc_cuda_load_param_vec_n(value off, value ex, value A, value v, value d){
	return spoc_cuda_load_param_vec(off, ex, A, v, d);
}

CAMLprim value spoc_cuda_custom_load_param_vec_b(value off, value ex, value A, value v){
	return spoc_cuda_custom_load_param_vec(off, ex, A, v);

}
CAMLprim value spoc_cuda_custom_load_param_vec_n(value off, value ex, value A, value v){
	return spoc_cuda_custom_load_param_vec(off, ex, A, v);
}

CAMLprim value spoc_cuda_load_param_int_b(value off, value ex, value val){
	return spoc_cuda_load_param_int(off, ex, val);

}
CAMLprim value spoc_cuda_load_param_int_n(value off, value ex, value val){
	return spoc_cuda_load_param_int(off, ex, val);
}


CAMLprim value spoc_cuda_load_param_int64_b(value off, value ex, value val){
	return spoc_cuda_load_param_int64(off, ex, val);

}
CAMLprim value spoc_cuda_load_param_int64_n(value off, value ex, value val){
	return spoc_cuda_load_param_int64(off, ex, val);
}


CAMLprim value spoc_cuda_load_param_float_b(value off, value ex, value val){
	return spoc_cuda_load_param_float(off, ex, val);

}
CAMLprim value spoc_cuda_load_param_float_n(value off, value ex, value val){
	return spoc_cuda_load_param_float(off, ex, val);
}


CAMLprim value spoc_cuda_load_param_float64_b(value off, value ex, value val){
	return spoc_cuda_load_param_float64(off, ex, val);

}
CAMLprim value spoc_cuda_load_param_float64_n(value off, value ex, value val){
	return spoc_cuda_load_param_float64(off, ex, val);
}

CAMLprim value spoc_cuda_set_block_shape(value ker, value block, value gi){
	CAMLparam3(ker, block, gi);
	CUfunction *kernel;

	CUDA_GET_CONTEXT;

	kernel = (CUfunction*) ker;
	CUDA_CHECK_CALL(cuFuncSetBlockShape(*kernel, Int_val(Field(block,0)),Int_val(Field(block,1)),Int_val(Field(block,2))));

	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_launch_grid(value off, value ker, value grid, value block, value ex, value gi, value queue_id){
  CAMLparam5(ker, grid, ex, block, gi);
  CAMLxparam2(off, queue_id);
  CUfunction *kernel;
  int gridX, gridY, gridZ, blockX, blockY, blockZ;
  int offset;
  char* extra;
  void* extra2[5];
  offset = Int_val(Field(off, 0));

  gridX = Int_val(Field(grid,0));
  gridY = Int_val(Field(grid,1));
  gridZ = Int_val(Field(grid,2));
  blockX = Int_val(Field(block,0));
  blockY = Int_val(Field(block,1));
  blockZ = Int_val(Field(block,2));
  
  CUDA_GET_CONTEXT;
  
  kernel = (CUfunction*) ker;  
  extra = (char*)ex;
  
  extra2[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
  extra2[1] = extra;
  extra2[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
  extra2[3] = &offset;
  extra2[4] = CU_LAUNCH_PARAM_END;
  
  
  CUDA_CHECK_CALL(cuLaunchKernel(*kernel, 
				 gridX, gridY, gridZ,
				 blockX, blockY, blockZ, 
				 0, queue[Int_val(queue_id)], 
				 NULL, extra2));

  Store_field(off, 0, Val_int(offset));
  free(extra);
  CUDA_RESTORE_CONTEXT;
  CAMLreturn(Val_unit);
}

CAMLprim value spoc_cuda_flush(value gi, value dev, value q){
	CAMLparam3(gi, dev, q);
	cuda_event_list *events;
	events =  (cuda_event_list*)(Field(dev,3));

	CUDA_GET_CONTEXT;

	cuStreamSynchronize(queue[Int_val(q)]);
	/*while(events != NULL && events->next != NULL)
	{
		if (events != NULL) {
			CUDA_CHECK_CALL(cuEventSynchronize (events->evt));
			if (events->vec)
			{
				cuMemFree (events->vec);
			}
			CUDA_CHECK_CALL(cuEventDestroy(events->evt));
		}
		{cuda_event_list *tmp= events;
		events = events->next;
		if (tmp != NULL)
			free(tmp);}
	}
	if (events) free(events);
	events = NULL;*/
	CUDA_RESTORE_CONTEXT;
	Store_field(dev, 3, (value)events);

	CAMLreturn(Val_unit);
}


CAMLprim value spoc_cuda_flush_all(value gi, value dev){
	CAMLparam2(gi, dev);

	CUDA_GET_CONTEXT;
	cuCtxSynchronize();
	CUDA_RESTORE_CONTEXT;

	CAMLreturn(Val_unit);
}

  CAMLprim value spoc_cuda_launch_grid_n(value off, value ker, value grid, value block,  value ex, value gi, value queue_id){
    return spoc_cuda_launch_grid(off, ker, grid, block,  ex, gi, queue_id);
    
  }
  
  CAMLprim value spoc_cuda_launch_grid_b(value* vv , int nb_vals)
  {
    value off = vv[0];
    value ker = vv[1];
    value grid = vv[2];
    value block = vv[3];
    value ex = vv[4];
    value gi = vv[5];
    value queue_id = vv[6];
    return spoc_cuda_launch_grid(off, ker, grid, block, ex, gi, queue_id);
  }
#ifdef __cplusplus
}
#endif
