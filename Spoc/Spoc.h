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
#ifndef _SPOC_H_
#define _SPOC_H_
#include <stdio.h>
#include <stdlib.h>

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <caml/signals.h>

#include "../dependencies/Cuda/cuComplex.h"



#include "cuda_drvapi_dynlink_cuda.h"
#include "Opencl_dynlink.h"

/*#ifdef _WIN32
#define THREAD_LOCAL   static  __declspec(thread)
#else
#define THREAD_LOCAL //static __thread
#endif
*/

/* for profiling */
//#define PROFILE


extern int noCuda;
extern int noCL;

int nbCudaDevices;

// should probably be acquired from device_info
#define OPENCL_PAGE_ALIGN 4096
#define OPENCL_CACHE_ALIGN 64

typedef struct cuda_device_ptrs {
	CUdeviceptr curr;
	size_t size;
	struct cuda_device_ptrs* next;
} device_ptrs;

typedef struct spoc_cuda_gc_info{
	CUdeviceptr start_ptr;
	CUdeviceptr end_ptr;
	CUdeviceptr curr_ptr;

	device_ptrs *used_ptrs;
	device_ptrs *unused_ptrs;

}
spoc_cuda_gc_info;

typedef struct host_vector{
  int type;
  int type_size;
  int size;
  void* vec;
} host_vector;

typedef struct spoc_vector {
	int device;
	void* spoc_vec;
	CUdeviceptr cuda_device_vec;
	cl_mem opencl_device_vec;
	int length;
	int type_size;
	int vec_id;
}spoc_vector;



//#define DEBUG 1

#ifdef _WIN32
#define FUNCTION __FUNCTION__
#else
#define FUNCTION __func__
#endif

#ifdef DEBUG
#define PRINT_FUNC \
	printf("IN: %s @ %d\n", FUNCTION,__LINE__ ); fflush(stdout);
#else
#define PRINT_FUNC \
	printf("IN: %s@ %d\n", FUNCTION,__LINE__ ); fflush(stdout);
#endif

#define RAISE_CUDA_ERROR \
	switch (cuda_error){ \
		case  CUDA_ERROR_DEINITIALIZED: \
			raise_constant(*caml_named_value("cuda_error_deinitialized")) ; \
			break; \
		case CUDA_ERROR_NOT_INITIALIZED: \
			raise_constant(*caml_named_value("cuda_error_not_initialized")) ; \
			break; \
		case CUDA_ERROR_INVALID_CONTEXT: \
			raise_constant(*caml_named_value("cuda_error_invalid_context")) ; \
			break; \
		case CUDA_ERROR_INVALID_VALUE: \
			raise_constant(*caml_named_value("cuda_error_invalid_value")) ; \
			break; \
		case CUDA_ERROR_OUT_OF_MEMORY: \
			raise_constant(*caml_named_value("cuda_error_out_of_memory")) ; \
			break; \
		case CUDA_ERROR_INVALID_DEVICE: \
			raise_constant(*caml_named_value("cuda_error_invalid_device")) ; \
			break; \
		case CUDA_ERROR_NOT_FOUND: \
			raise_constant(*caml_named_value("cuda_error_not_found")) ; \
			break; \
		case CUDA_ERROR_FILE_NOT_FOUND: \
			raise_constant(*caml_named_value("cuda_error_file_not_found")) ; \
			break; \
		case CUDA_ERROR_LAUNCH_FAILED: \
			raise_constant(*caml_named_value("cuda_error_launch_failed")) ; \
			break; \
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: \
			raise_constant(*caml_named_value("cuda_error_launch_out_of_resources")) ; \
			break; \
		case CUDA_ERROR_LAUNCH_TIMEOUT: \
			raise_constant(*caml_named_value("cuda_error_launch_timeout")) ; \
			break; \
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: \
			raise_constant(*caml_named_value("cuda_error_launch_incompatible_texturing")) ; \
			break; \
		case  CUDA_ERROR_INVALID_HANDLE: \
/* ->*/			raise_constant(*caml_named_value("cuda_error_launch_incompatible_texturing")) ; \
			break; \
		case  CUDA_ERROR_ALREADY_MAPPED: \
/* ->*/			raise_constant(*caml_named_value("cuda_error_launch_incompatible_texturing")) ; \
			break; \
		case CUDA_ERROR_UNKNOWN: \
		default: \
			raise_constant(*caml_named_value("cuda_error_unknown")) ; \
			break;\
	}\


#define CUDA_CHECK_CALL(func) \
		cuda_error = func; \
		if (CUDA_SUCCESS != cuda_error ) { \
			printf("IN: %s@ %d\n", FUNCTION,__LINE__ ); fflush(stdout); \
			RAISE_CUDA_ERROR\
		}

#define ALIGN_UP(offset, alignment) \
       (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

#define UNIMPLENTED \
		printf("ERROR ************* %s unimplented\n", FUNCTION); \
		fflush(stdout);




#define OPENCL_TRY(name,fun)						\
  {									\
    cl_int err = fun;							\
    if (err != CL_SUCCESS) {						\
      fprintf(stderr,"ERROR %d calling %s().\n", err,name);             \
    }                                                                   \
  }

#define OPENCL_TRY2(name,fun, err)						\
  {									\
    err = fun;							\
    if (err != CL_SUCCESS) {						\
      fprintf(stderr,"ERROR %d calling %s().\n", err,name);             \
    }                                                                   \
  }


#define CL_FILE_NOT_FOUND -1

#define RAISE_OPENCL_ERROR \
	switch (opencl_error){ \
		case CL_INVALID_CONTEXT: raise_constant(*caml_named_value("opencl_invalid_context")) ; \
			break; \
		case CL_INVALID_DEVICE: raise_constant(*caml_named_value("opencl_invalid_device")) ; \
			break; \
		case CL_INVALID_VALUE: raise_constant(*caml_named_value("opencl_invalid_value")) ; \
			break; \
		case CL_INVALID_QUEUE_PROPERTIES: raise_constant(*caml_named_value("opencl_invalid_queue_properties")) ; \
			break; \
		case CL_OUT_OF_RESOURCES: raise_constant(*caml_named_value("opencl_out_of_resources")) ; \
			break; \
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:	raise_constant(*caml_named_value("opencl_mem_object_allocation_failure")) ; \
			break; \
		case CL_OUT_OF_HOST_MEMORY: raise_constant(*caml_named_value("opencl_out_of_host_memory")) ; \
			break; \
		case CL_INVALID_PROGRAM: raise_constant(*caml_named_value("opencl_invalid_program")) ; \
			break; \
		case CL_INVALID_BINARY: raise_constant(*caml_named_value("opencl_invalid_binary")) ; \
			break; \
		case CL_INVALID_BUILD_OPTIONS: raise_constant(*caml_named_value("opencl_invalid_build_options")) ; \
			break; \
		case CL_INVALID_OPERATION: raise_constant(*caml_named_value("opencl_invalid_operation")) ; \
			break; \
		case CL_COMPILER_NOT_AVAILABLE: raise_constant(*caml_named_value("opencl_compiler_not_available")) ; \
			break; \
		case CL_BUILD_PROGRAM_FAILURE: raise_constant(*caml_named_value("opencl_build_program_failure")) ; \
			break; \
		case CL_INVALID_KERNEL: raise_constant(*caml_named_value("opencl_invalid_kernel")) ; \
			break; \
		case CL_INVALID_ARG_INDEX: raise_constant(*caml_named_value("opencl_invalid_arg_index")) ; \
			break; \
		case CL_INVALID_ARG_VALUE: raise_constant(*caml_named_value("opencl_invalid_arg_value")) ; \
			break; \
		case CL_INVALID_MEM_OBJECT: raise_constant(*caml_named_value("opencl_invalid_mem_object")) ; \
			break; \
		case CL_INVALID_SAMPLER: raise_constant(*caml_named_value("opencl_invalid_sampler")) ; \
			break; \
		case CL_INVALID_ARG_SIZE: raise_constant(*caml_named_value("opencl_invalid_arg_size")) ; \
			break; \
		case CL_FILE_NOT_FOUND: raise_constant(*caml_named_value("opencl_file_not_found")) ; \
			break; \
		case CL_INVALID_COMMAND_QUEUE: raise_constant(*caml_named_value("opencl_invalid_command_queue")) ; \
			break; \
		default: raise_constant(*caml_named_value("opencl_error_unknown")) ; \
			break; \
}



#define OPENCL_CHECK_CALL1(val, fun) \
		val = fun; \
		if (CL_SUCCESS != opencl_error) { \
			printf("IN: %s@ %d\n", FUNCTION,__LINE__ ); fflush(stdout); \
			RAISE_OPENCL_ERROR \
		}



#define CUDA_GET_CONTEXT \
	{CUcontext ctx; \
	spoc_cu_context *spoc_ctx; \
	CUstream queue[2]; \
	enum cudaError_enum cuda_error = 0; \
	spoc_ctx = (spoc_cu_context*)Field(gi, 8); \
	ctx = spoc_ctx->ctx; \
	queue[0] = spoc_ctx->queue[0];\
	queue[1] = spoc_ctx->queue[1];\
	CUDA_CHECK_CALL(cuCtxSetCurrent(ctx)); \
	caml_enter_blocking_section();

#define BLOCKING_CUDA_GET_CONTEXT \
	{CUcontext ctx; \
	spoc_cu_context *spoc_ctx; \
	CUstream queue[2]; \
	enum cudaError_enum cuda_error = 0; \
	spoc_ctx = (spoc_cu_context*)Field(gi, 8); \
	ctx = spoc_ctx->ctx; \
	queue[0] = spoc_ctx->queue[0];\
	queue[1] = spoc_ctx->queue[1];\
	CUDA_CHECK_CALL(cuCtxSetCurrent(ctx));

#define CUDA_RESTORE_CONTEXT \
  caml_leave_blocking_section();		\
  spoc_ctx->queue[0] = queue[0]; \
  spoc_ctx->queue[1] = queue[1]; \
  Store_field(gi,8, (value)spoc_ctx);}

#define BLOCKING_CUDA_RESTORE_CONTEXT \
  spoc_ctx->queue[0] = queue[0]; \
  spoc_ctx->queue[1] = queue[1]; \
  Store_field(gi,8, (value)spoc_ctx);}

typedef struct spoc_cl_context {
		cl_context ctx;
		cl_command_queue queue[2];
}spoc_cl_context;

typedef struct spoc_cu_context {
		CUcontext ctx;
		CUstream queue[2];
}spoc_cu_context;

#define OPENCL_GET_CONTEXT \
	{cl_context ctx; \
	cl_command_queue queue[2]; \
	spoc_cl_context *spoc_ctx; \
	cl_int opencl_error = 0; \
	spoc_ctx = (spoc_cl_context*)Field(gi, 8); \
	ctx = spoc_ctx->ctx; \
	queue[0] = spoc_ctx->queue[0];\
	queue[1] = spoc_ctx->queue[1];\
	caml_enter_blocking_section();



#define OPENCL_RESTORE_CONTEXT \
	caml_leave_blocking_section();		\
	spoc_ctx->queue[0] = queue[0]; \
	spoc_ctx->queue[1] = queue[1]; \
	spoc_ctx->ctx = ctx;\
	Store_field(gi,8, (value)spoc_ctx);}


int ae_load_file_to_memory(const char *filename, char **result);




#define GET_TYPE_SIZE				\
  if (custom) {								\
    type_size = Int_val(Field(Field(bigArray, 1),0))*sizeof(char);	\
  }									\
  else {								\
    type_size=((host_vector*)(Field(Field(bigArray,0),1)))->type_size;	\
  }
/* #define GET_TYPE_SIZE 						  \ */
/*   if (custom) { \ */
/*   type_size = Int_val(Field(Field(bigArray, 1),0))*sizeof(char); \ */
/*   }								 \ */
/*   else { \ */
/*   tag = Bigarray_val(bigArray)->flags & BIGARRAY_KIND_MASK;	\ */
/* 	switch (tag) { \ */
/* 	case BIGARRAY_UINT8 : \ */
/* 		printf("here\n"); \ */
/* 		type_size = sizeof(unsigned char); \ */
/* 		break; \ */
/* 	case BIGARRAY_FLOAT32 : \ */
/* 		type_size = sizeof(float); \ */
/* 		break; \ */
/* 	case BIGARRAY_FLOAT64 : \ */
/* 		type_size = sizeof(double); \ */
/* 		break; \ */
/* 	case BIGARRAY_INT32 : \ */
/* 		type_size = sizeof(int); \ */
/* 		break; \ */
/* 	case BIGARRAY_INT64 : \ */
/* 		type_size = sizeof(int)*2; \ */
/* 		break; \ */
/* 	case BIGARRAY_COMPLEX32: \ */
/* 		type_size = sizeof(float)*2;\ */
/* 		break; \ */
/* 		} \ */
/* 	} */

#define Val_cl_mem(x) (value)((cl_mem)x)
#define Cl_mem_val(x) (cl_mem)((value)x)

#define Val_CUdeviceptr(x) (value)((CUdeviceptr)x)
#define CUdeviceptr_val(x) (CUdeviceptr)((value)x)

cuComplex complex_val (value c);
cuDoubleComplex doubleComplex_val (value c);
value copy_two_doubles(double d0, double d1);
value copy_complex(cuComplex c);
value copy_doubleComplex(cuDoubleComplex c);

#define Complex_val(c) complex_val(c)

typedef struct vector_list {
		spoc_vector current;
		struct vector_list* next;
}vector_list;

typedef struct dev_vectors {
	void* dev;
	vector_list* vectors;
	struct dev_vectors* next;
}dev_vectors;


typedef struct cuda_event_list {
	CUevent evt;
	CUdeviceptr vec;
	struct cuda_event_list *next;
}cuda_event_list;



typedef struct spoc_cl_vector {
	long cu_size;
	cl_mem cu_vector;
}cl_vector;

typedef struct spoc_cu_vector {
	long cu_size;
	CUdeviceptr cu_vector;
}cu_vector;


#endif
