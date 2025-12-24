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
  #include "cuda_drvapi_dynlink_cuda.h"
  int noCuda = 1;

  #include "Spoc.h"

/* OCaml 5 compatible: custom block for CUDA context */
static void cuda_context_finalize(value v) {
    spoc_cu_context* ctx = *((spoc_cu_context**)Data_custom_val(v));
    if (ctx) {
        if (ctx->queue[0]) cuStreamDestroy(ctx->queue[0]);
        if (ctx->queue[1]) cuStreamDestroy(ctx->queue[1]);
        if (ctx->ctx) cuCtxDestroy(ctx->ctx);
        free(ctx);
        *((spoc_cu_context**)Data_custom_val(v)) = NULL;
    }
}

static struct custom_operations cuda_context_ops = {
    .identifier = "spoc.cuda_context",
    .finalize = cuda_context_finalize,
    .compare = custom_compare_default,
    .hash = custom_hash_default,
    .serialize = custom_serialize_default,
    .deserialize = custom_deserialize_default,
    .compare_ext = custom_compare_ext_default,
    .fixed_length = custom_fixed_length_default
};

#define Cuda_context_val(v) (*((spoc_cu_context**)Data_custom_val(v)))
#define Set_cuda_context(v, x) (*((spoc_cu_context**)Data_custom_val(v)) = (x))

/* OCaml 5 compatible: custom block for CUDA GC info */
static void cuda_gc_info_finalize(value v) {
    spoc_cuda_gc_info* info = *((spoc_cuda_gc_info**)Data_custom_val(v));
    if (info) {
        free(info);
        *((spoc_cuda_gc_info**)Data_custom_val(v)) = NULL;
    }
}

static struct custom_operations cuda_gc_info_ops = {
    .identifier = "spoc.cuda_gc_info",
    .finalize = cuda_gc_info_finalize,
    .compare = custom_compare_default,
    .hash = custom_hash_default,
    .serialize = custom_serialize_default,
    .deserialize = custom_deserialize_default,
    .compare_ext = custom_compare_ext_default,
    .fixed_length = custom_fixed_length_default
};

#define Cuda_gc_info_val(v) (*((spoc_cuda_gc_info**)Data_custom_val(v)))
#define Set_cuda_gc_info(v, x) (*((spoc_cuda_gc_info**)Data_custom_val(v)) = (x))



  
  value spoc_cuInit() {

    CAMLparam0();
    if ((CUDA_SUCCESS != cuInit(0, 4000)))
      /*		&& (CUDA_SUCCESS != cuInit(0, 3020))
			&& (CUDA_SUCCESS != cuInit(0, 3010))
			&& (CUDA_SUCCESS != cuInit(0, 3000))
			&& (CUDA_SUCCESS != cuInit(0, 2030))
			&& (CUDA_SUCCESS != cuInit(0, 2010))
			&& (CUDA_SUCCESS != cuInit(0, 2000))
			&& (CUDA_SUCCESS != cuInit(0, 1000)))*/
      {
	//noCuda = 1;
	CAMLreturn(Val_unit);
      }
    noCuda=0;
    CAMLreturn(Val_unit);
  }

  dev_vectors *usefull_vectors;
  dev_vectors *useless_vectors;

  void add_usefull_vector (spoc_vector v, void* dev) {
    dev_vectors *vecs = usefull_vectors;
    while (vecs)
      {
	if (vecs->dev == dev)
	  {
	    vector_list *vlist = vecs->vectors;
	    while (vlist)
	      {
		if (vlist->current.vec_id == v.vec_id)
		  {
		    //do nothing
		    return;
		  }
		else vlist = vlist->next;
	      }
	    {vector_list* new_vectors = malloc(sizeof(vector_list));
	      new_vectors->next = vecs->vectors;
	      new_vectors->current = v;
	      vecs->vectors= new_vectors;
	      return;}
	  }
	else
	  vecs = vecs->next;
      }
    {dev_vectors* new_usefull = malloc(sizeof(dev_vectors));
      new_usefull->next = usefull_vectors;
      new_usefull->vectors= malloc(sizeof(vector_list));
      new_usefull->vectors->current = v;
      new_usefull->vectors->next = NULL;
      new_usefull->dev = dev;
      usefull_vectors = new_usefull;
      return;}
  }

  void remove_usefull_vector (spoc_vector v, void* dev){
    dev_vectors *vecs = usefull_vectors;
    while (vecs)
      {
	if (vecs->dev == dev)
	  {
	    vector_list *vlist = vecs->vectors;
	    while (vlist)
	      {
		if (vlist->current.vec_id == v.vec_id)
		  {
		    if (vlist->next)
		      {
			vector_list* tmp = vlist->next;
			free(vlist);
			vlist = tmp;
		      }
		    else free(vlist);
		    return;
		  }
		else vlist = vlist->next;
	      }
	    return;
	  }
	else
	  vecs = vecs->next;
      }
    return;
  }

  value spoc_getCudaDevicesCount()
  {
    CAMLparam0();
    int nb_devices;
    if (noCuda) CAMLreturn (Val_int(0));
    cuDeviceGetCount (&nb_devices);
    nbCudaDevices = nb_devices;
    CAMLreturn (Val_int(nb_devices));

  }




  value spoc_getCudaDevice(value i)
  {
    CAMLparam1(i);
    CAMLlocal4(general_info, cuda_info, specific_info, gc_info);
    CAMLlocal3(device,  maxT, maxG);
    int nb_devices;
    CUdevprop dev_infos;
    CUdevice dev;
    CUcontext ctx;
    CUstream queue[2];
    spoc_cu_context *spoc_ctx;
    //CUcontext gl_ctx;
    char infoStr[1024];
    int infoInt;
    size_t infoUInt;
    int major, minor;
    enum cudaError_enum cuda_error;


    cuDeviceGetCount (&nb_devices);

    if ((Int_val(i)) > nb_devices)
      caml_raise_constant(*caml_named_value("no_cuda_device")) ;


    CUDA_CHECK_CALL(cuDeviceGet(&dev, Int_val(i)));
    CUDA_CHECK_CALL(cuDeviceGetProperties(&dev_infos, dev));

    general_info = caml_alloc (9, 0);
    CUDA_CHECK_CALL(cuDeviceGetName(infoStr, sizeof(infoStr), dev));

    Store_field(general_info,0, caml_copy_string(infoStr));//
    CUDA_CHECK_CALL(cuDeviceTotalMem(&infoUInt, dev));

    Store_field(general_info,1, Val_int(infoUInt));//
    Store_field(general_info,2, Val_int(dev_infos.sharedMemPerBlock));//
    Store_field(general_info,3, Val_int(dev_infos.clockRate));//
    Store_field(general_info,4, Val_int(dev_infos.totalConstantMemory));//
    CUDA_CHECK_CALL(cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    Store_field(general_info,5, Val_int(infoInt));//
    CUDA_CHECK_CALL(cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, dev));
    Store_field(general_info,6, Val_bool(infoInt));//
    Store_field(general_info,7, i);
    CUDA_CHECK_CALL(cuCtxCreate	(&ctx,
				 CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST,
				 dev));
    spoc_ctx = malloc(sizeof(spoc_cu_context));
    spoc_ctx->ctx = ctx;
    CUDA_CHECK_CALL(cuStreamCreate(&queue[0], 0));
    CUDA_CHECK_CALL(cuStreamCreate(&queue[1], 0));
    spoc_ctx->queue[0] = queue[0];
    spoc_ctx->queue[1] = queue[1];
    /* OCaml 5: store context in a custom block */
    {
        value ctx_block = caml_alloc_custom(&cuda_context_ops, sizeof(spoc_cu_context*), 0, 1);
        Set_cuda_context(ctx_block, spoc_ctx);
        Store_field(general_info, 8, ctx_block);
    }
    CUDA_CHECK_CALL(cuCtxSetCurrent(ctx));


    cuda_info = caml_alloc(1, 0); //0 -> Cuda
    specific_info = caml_alloc(18, 0);

    cuDeviceComputeCapability(&major, &minor, dev);
    Store_field(specific_info,0, Val_int(major));//
    Store_field(specific_info,1, Val_int(minor));//
    Store_field(specific_info,2, Val_int(dev_infos.regsPerBlock));//
    Store_field(specific_info,3, Val_int(dev_infos.SIMDWidth));//
    Store_field(specific_info,4, Val_int(dev_infos.memPitch));//
    Store_field(specific_info,5, Val_int(dev_infos.maxThreadsPerBlock));//

    maxT = caml_alloc(3, 0);
    Store_field(maxT,0, Val_int(dev_infos.maxThreadsDim[0]));//
    Store_field(maxT,1, Val_int(dev_infos.maxThreadsDim[1]));//
    Store_field(maxT,2, Val_int(dev_infos.maxThreadsDim[2]));//
    Store_field(specific_info,6, maxT);

    maxG = caml_alloc(3, 0);
    Store_field(maxG,0, Val_int(dev_infos.maxGridSize[0]));//
    Store_field(maxG,1, Val_int(dev_infos.maxGridSize[1]));//
    Store_field(maxG,2, Val_int(dev_infos.maxGridSize[2]));//
    Store_field(specific_info,7, maxG);

    Store_field(specific_info,8, Val_int(dev_infos.textureAlign));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, dev);
    Store_field(specific_info,9, Val_bool(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, dev);
    Store_field(specific_info,10, Val_bool(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev);
    Store_field(specific_info,11, Val_bool(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev);
    Store_field(specific_info,12, Val_bool(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev);
    Store_field(specific_info,13, Val_int(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, dev);
    Store_field(specific_info,14, Val_bool(infoInt));//
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev);
    Store_field(specific_info,15, Val_int(infoInt));
    cuDeviceGetAttribute(&infoInt, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev);
    Store_field(specific_info,16, Val_int(infoInt));
    cuDriverGetVersion(&infoInt);
    Store_field(specific_info, 17, Val_int(infoInt));

    Store_field(cuda_info, 0, specific_info);
    device = caml_alloc(4, 0);
    Store_field(device, 0, general_info);
    Store_field(device, 1, cuda_info);

    {spoc_cuda_gc_info* gcInfo = (spoc_cuda_gc_info*)malloc(sizeof(spoc_cuda_gc_info));
      CUDA_CHECK_CALL(cuMemGetInfo(&infoUInt, NULL));
      infoUInt -= (32*1024*1024);

      /* OCaml 5: store gc_info in a custom block */
      {
          value gc_block = caml_alloc_custom(&cuda_gc_info_ops, sizeof(spoc_cuda_gc_info*), 0, 1);
          Set_cuda_gc_info(gc_block, gcInfo);
          Store_field(device, 2, gc_block);
      }

      /* events can be NULL initially, store as Val_int(0) which is safe */
      Store_field(device, 3, Val_int(0));

      CAMLreturn(device);}
  }

#ifdef __cplusplus
}
#endif
