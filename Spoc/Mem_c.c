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
  
#include "Mem_c.h"
#include "Trac_c.h"
#include <assert.h>


#define spocsizeof(name,typename)               \
  CAMLprim value sizeof##name(){                \
    CAMLparam0();                               \
    CAMLreturn (Val_int(sizeof(typename)));     \
  }						
  
  spocsizeof(Float32,float)
  spocsizeof(Float64,double)
  spocsizeof(Int32,int)
  spocsizeof(Int64,long)
  spocsizeof(Char,char)
  spocsizeof(Complex32,double2)
  
#ifdef PROFILE
#define  get(name,type,macro)                       \
  CAMLprim value get_##name(value v, value idx){	\
    CAMLparam2(v, idx);                             \
    host_vector* vec = (host_vector*) (Field(v,1));	\
    type r = ((type*)(vec->vec))[Int_val(idx)];		\
    print_get_vector(Field(v, 9), Field(v, 0));     \
    CAMLreturn (macro(r));                          \
  }

#define  set(name,type,macro)                                   \
  CAMLprim value set_##name(value v, value idx, value val ){	\
    CAMLparam3(v, idx, val);                                    \
    host_vector* vec = (host_vector*) (Field(v,1));             \
    type r = macro(val);                                        \
    ((type*)(vec->vec))[Int_val(idx)] = r;                      \
    print_set_vector(Field(v, 9), Field(v, 0));                 \
    CAMLreturn (Val_unit);                                      \
  }

#else

#define  get(name,type,macro)                       \
  CAMLprim value get_##name(value v, value idx){	\
    CAMLparam2(v, idx);                             \
    host_vector* vec = (host_vector*) (Field(v,1));	\
    type r = ((type*)(vec->vec))[Int_val(idx)];		\
    CAMLreturn (macro(r));                          \
  }

#define  set(name,type,macro)                                   \
  CAMLprim value set_##name(value v, value idx, value val ){	\
    CAMLparam3(v, idx, val);                                    \
    host_vector* vec = (host_vector*) (Field(v,1));             \
    type r = macro(val);                                        \
    ((type*)(vec->vec))[Int_val(idx)] = r;                      \
    CAMLreturn (Val_unit);                                      \
  }
#endif  


  get(int32,int,caml_copy_int32)
  get(int64,long,caml_copy_int64)
  get(float64,double,caml_copy_double)
  get(char,int,Val_int)

  set(int32,int,Int32_val)
  set(int64,long,Int64_val)
  set(float64,double,Double_val)
  set(char,int,Int_val)
  
  CAMLprim value get_float32(value v, value idx){	
    CAMLparam2(v, idx);					
    host_vector* vec = (host_vector*) (Field(v,1));	
    float r = ((float*)(vec->vec))[Int_val(idx)];		
    CAMLreturn (caml_copy_double((double)r));				
  }


  CAMLprim value set_float32(value v, value idx, value val ){	
    CAMLparam3(v, idx, val);					
    host_vector* vec = (host_vector*) (Field(v,1));
    float r = (float)Double_val(val);					
    ((float*)(vec->vec))[Int_val(idx)] = r;
    CAMLreturn (Val_unit);				       
    }
  
  CAMLprim value get_complex32(value v, value idx){
    CAMLparam2(v, idx);					
    UNIMPLENTED;
    CAMLreturn(Val_unit);
  }

  CAMLprim value set_complex32(value v, value idx, value val ){	
    CAMLparam3(v, idx, val);
    UNIMPLENTED;
    CAMLreturn (Val_unit);
  }
  
  void free_host(value v){
    host_vector* vec = (host_vector*) (Field(v,1));
    if (vec->vec) {
      if  (noCuda){
        free(vec->vec);
      }
      else{
        cuMemFreeHost(vec->vec);
      }
    }
    free(vec);
  }
  
  CAMLprim value host_alloc (value type_size, value n){
    CAMLparam2(type_size,n);
    CAMLlocal1(ret);
    ret=alloc_final(2, free_host, 0,  1);
    host_vector* v= (host_vector*)malloc(sizeof(host_vector));
    v->type_size=Int_val(type_size);
    v->size=Int_val(n);
    if (noCuda){
      posix_memalign(&(v->vec), OPENCL_PAGE_ALIGN,
                     ((Int_val(type_size)*Int_val(n) - 1)/OPENCL_CACHE_ALIGN + 1) * 
                     OPENCL_CACHE_ALIGN);
    }
    else{
      cuMemAllocHost(&(v->vec), Int_val(type_size)*Int_val(n));
    }
    Store_field(ret,1,v);
    CAMLreturn(ret);
  }

  CAMLprim value spoc_bigarray_adress(value ba, value type_size, value n){
    CAMLparam3(ba, type_size,n);
    CAMLlocal1(ret);
    ret=caml_alloc(2,0);
    host_vector* v= (host_vector*)malloc(sizeof(host_vector));
    v->type_size=Int_val(type_size);
    v->size=Int_val(n);
    v->vec=Caml_ba_data_val(ba);
    Store_field(ret,1,v);
    CAMLreturn (ret);
  }


  CAMLprim value spoc_sub_host_vec(value host_vec, value start, value len){
    CAMLparam3(host_vec, start, len);
    CAMLlocal1(ret);
    ret=caml_alloc(2,0);
    host_vector* parent = (host_vector*) (Field(host_vec,1));
    host_vector* v = (host_vector*)malloc(sizeof(host_vector));
    v->type_size = parent->type_size;
    v->size = len;
    v->vec = (char*)parent->vec+(v->type_size*Int_val(start));
    Store_field(ret,1,v);
    CAMLreturn(ret);
  }

  
  void cuda_free_vec (value v) {
    cu_vector* cuv = (cu_vector*)(Field(v, 1));
    if (cuv)
      {
	CUdeviceptr f = cuv->cu_vector;
	
#ifdef PROFILE
      value bigArray;
      int size;
      int type_size;
      int seek;
      int tag;
      bigArray = Field (Field(v, 1), 0);
      seek = Int_val(Field(v, 10));
      size = Int_val(Field(v, 4))-seek;
      int custom = 0;
      GET_TYPE_SIZE;
      print_gpu_free("GPU_FREE", Field(v, 9), Field(v,0), "CUDA", size*type_size);
#endif
      enum cudaError_enum cuda_error = 0;
      if (f)
	  {
	    cuMemFree(f);
	    free (cuv);
	  }
    }
  }


  CAMLprim value spoc_finalize_vector (value bigArray){
    CAMLparam1(bigArray);
    CUdevice dev = 0;
    size_t size_left = 0;
    enum cudaError_enum cuda_error = 0;
    struct caml_ba_array * b = Caml_ba_array_val(bigArray);
    int size = caml_ba_byte_size(b);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_init_cuda_device_vec(){
    CAMLparam0();
    CAMLlocal1(ret);
    ret = alloc_final(2, cuda_free_vec, 0, 1);
    Store_field(ret, 1, (value)NULL);
    CAMLreturn(ret);
  }

  void cl_free_vec (value v) {
    cl_mem f = Cl_mem_val(Field(v, 1));
    if (f) {
      clReleaseMemObject(f);
      Store_field(v,1,NULL);
    }
  }

  CAMLprim value spoc_init_opencl_device_vec(){
    CAMLparam0();
    CAMLlocal1(ret);
    ret = alloc_final(2, cl_free_vec, 0, 1);
    Store_field(ret, 1, (value)NULL);

    CAMLreturn(ret);
  }

  CAMLprim value spoc_cuda_part_cpu_to_device(value vector, value sub_vector, value nb_device, 
                                              value gi, value gc_info, value queue_id, 
                                              value host_offset, value guest_offset, 
                                              value start, value part_size){
    CAMLparam5(vector, nb_device, gi, gc_info, queue_id);
    CAMLxparam5(sub_vector, host_offset, guest_offset, start, part_size);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    char* h_A = 0;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int tag;
    int seek;

    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(sub_vector, 2);
    dev_vec = Field(dev_vec_array, Int_val(nb_device));
    seek = Int_val(Field(vector, 10));
    CUDA_GET_CONTEXT;
    int custom = 0;
    GET_TYPE_SIZE;

    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec)+
      ((Long_val(host_offset)+Long_val(start)+seek)*type_size);
	//(void*)((char*)Data_bigarray_val(bigArray)+((Long_val(host_offset)+Long_val(start)+seek)*type_size));

    size = Int_val(Field(sub_vector, 4))-seek;

#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;    
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
#endif

    CUDA_CHECK_CALL(cuMemcpyHtoDAsync(d_A+((Long_val(guest_offset)+seek)*type_size), h_A, 
                                      (Long_val(part_size))*type_size, queue[Int_val(queue_id)]));

#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    print_start_part_transfert("PART_CPU_TO_DEVICE", ((Long_val(part_size))*type_size), 
                               ((Int_val(Field(vector, 4))-seek)*type_size),
                               Int_val(Field(sub_vector, 9)), Int_val(Field(vector, 9)), 
                               "CUDA", Int_val(nb_device), NULL, events);
#endif

    CUDA_RESTORE_CONTEXT;
    //Store_field(dev_vec, 1, Val_CUdeviceptr(d_A));
    //Store_field(dev_vec_array, Int_val(nb_device), dev_vec);
    //Store_field(sub_vector, 2, dev_vec_array);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_part_cpu_to_device_b(value *tab_val, value nb_val){
    value vector = tab_val[0];
    value sub_vector = tab_val[1];
    value nb_device = tab_val[2];
    value gi = tab_val[3];
    value gc_info = tab_val[4];
    value queue_id = tab_val[5];
    value host_offset = tab_val[6];
    value guest_offset = tab_val[7];
    value start = tab_val[8];
    value part_size = tab_val[9];
    return spoc_cuda_part_cpu_to_device(vector, sub_vector, nb_device, gi, gc_info, 
                                        queue_id, host_offset, guest_offset,  start, part_size);
  }

  CAMLprim value spoc_cuda_part_cpu_to_device_n(value vector, value subvector, value nb_device, 
                                                value gi, value gc_info, value queue_id, 
                                                value host_offset, value guest_offset, value start,
                                                value part_size){
    return spoc_cuda_part_cpu_to_device(vector, subvector, nb_device, gi, gc_info, 
                                        queue_id, host_offset, guest_offset, start, part_size);
  }



  CAMLprim value spoc_cuda_cpu_to_device(value vector, value nb_device, value gi, value gc_info,
                                         value queue_id){
    CAMLparam5(vector, nb_device, gi, gc_info, queue_id);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int tag;
    int seek;

    spoc_cuda_gc_info* gcInfo;

    seek = Int_val(Field(vector, 10));
    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(vector, 2);
    dev_vec = Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;
    CUDA_GET_CONTEXT;
    gcInfo = (spoc_cuda_gc_info*) gc_info;
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec);
  //h_A = (void*)Data_bigarray_val(bigArray);
    int custom = 0;
    GET_TYPE_SIZE;
    size =Int_val(Field(vector, 4))-seek;
    
#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;    
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
#endif

    CUDA_CHECK_CALL(cuMemcpyHtoDAsync(d_A+(seek*type_size), h_A+(seek*type_size), 
                                      size*type_size, queue[Int_val(queue_id)]));

#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    print_start_transfert("CPU_TO_DEVICE", size*type_size, 
                          Int_val(Field(vector, 9)), "CUDA", Int_val(Field(gi, 7)),
                          NULL, events);
#endif

    //Store_field(dev_vec, 1, Val_CUdeviceptr(d_A));
    //Store_field(dev_vec, 2, (value) spoc_ctx);
    //Store_field(dev_vec_array, Int_val(nb_device), dev_vec);
    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_custom_part_cpu_to_device(value vector, value sub_vector, 
                                                     value nb_device, value gi, value queue_id,
                                                     value offset, value start, value part_size){
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLxparam4(sub_vector, offset, start, part_size);
    CAMLlocal5(device_vec, vec, customArray,  dev_vec_array, dev_vec);
    char* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int tag;
    int seek;

    customArray = Field (Field(vector, 1), 0);
    h_A = (char*)Field(Field(customArray, 0),1);
    seek = Int_val(Field(vector, 10));

    type_size = Int_val(Field(Field(customArray, 1),0));
    size =Int_val(Field(vector, 4))-seek;

    dev_vec_array = Field(sub_vector, 2);
    dev_vec = Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;

    CUDA_GET_CONTEXT;
#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;    
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
#endif
    CUDA_CHECK_CALL(cuMemcpyHtoDAsync(d_A+(Int_val(offset)+seek*type_size), 
                                      h_A+(Int_val(offset))+Int_val(start)+seek*type_size, 
                                      part_size*type_size, queue[Int_val(queue_id)]));
#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    print_start_part_transfert("PART_CPU_TO_DEVICE_CUSTOM", ((Long_val(part_size))*type_size),
                               ((Int_val(Field(vector, 4))-seek)*type_size), 
                               Int_val(Field(sub_vector, 9)), Int_val(Field(vector, 9)),
                               "CUDA", Int_val(nb_device), NULL, events);
#endif

    //Store_field(dev_vec, 1, Val_CUdeviceptr(d_A));
    //Store_field(dev_vec, 2, (value) spoc_ctx);
    //Store_field(dev_vec_array, Int_val(nb_device), dev_vec);
    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }
  CAMLprim value spoc_cuda_custom_part_cpu_to_device_b(value vector, value sub_vector, 
                                                       value nb_device, value gi, value queue_id,
                                                       value offset, value start, value part_size)
  {
    return spoc_cuda_custom_part_cpu_to_device(vector, sub_vector, nb_device, gi, queue_id, 
                                               offset, start, part_size);
  }
  CAMLprim value spoc_cuda_custom_part_cpu_to_device_n(value vector, value sub_vector, 
                                                       value nb_device, value gi, value queue_id,
                                                       value offset, value start, value part_size){
    return spoc_cuda_custom_part_cpu_to_device(vector, sub_vector, nb_device, gi, queue_id, 
                                               offset, start, part_size);
  }
  
  CAMLprim value spoc_opencl_custom_part_cpu_to_device_b(value vector, value sub_vector, 
                                                         value nb_device, value gi, value queue_id,
                                                         value offset, value part_size){
    return spoc_cuda_custom_part_cpu_to_device(vector, sub_vector, nb_device, gi, queue_id, 
                                               offset, 0, part_size);
  }
  CAMLprim value spoc_opencl_custom_part_cpu_to_device_n(value vector, value sub_vector, 
                                                         value nb_device, value gi, value queue_id,
                                                         value offset, value part_size){
    return spoc_cuda_custom_part_cpu_to_device(vector, sub_vector, nb_device, gi, queue_id, 
                                               offset, 0, part_size);
  }

  CAMLprim value spoc_cuda_custom_cpu_to_device(value vector, value nb_device, value gi, 
                                                value queue_id){
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLlocal5(device_vec, vec, customArray,  dev_vec_array, dev_vec);
    char* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int tag;
    int seek;

    customArray = Field (Field(vector, 1), 0);
    h_A = (char*)Field(Field(customArray, 0),1);
    seek = Int_val(Field(vector, 10));

    type_size = Int_val(Field(Field(customArray, 1),0));
    size =Int_val(Field(vector, 4))-seek;

    dev_vec_array = Field(vector, 2);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;

    CUDA_GET_CONTEXT;
    
#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;  
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
#endif

    CUDA_CHECK_CALL(cuMemcpyHtoDAsync(d_A+seek*type_size, h_A+(seek*type_size), size*type_size, queue[Int_val(queue_id)]));

#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    print_start_transfert("CPU_TO_DEVICE_CUSTOM", size*type_size, Int_val(Field(vector, 9)), 
                          "CUDA", Int_val(Field(gi, 7)), NULL, events);
#endif

    //Store_field(dev_vec, 1, Val_CUdeviceptr(d_A));
    //Store_field(dev_vec, 2, (value) spoc_ctx);
    //Store_field(dev_vec_array, Int_val(nb_device), dev_vec);
    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }



  CAMLprim value spoc_cuda_device_to_device() {
    CAMLparam0();
    UNIMPLENTED;

    CAMLreturn(Val_unit);

  }


  void cuda_free_after_transfer(CUstream stream, CUresult status, void* data) {
#ifdef PROFILE
    value dev_vec;
    value dev_vec_array;
    value bigArray;
    value* tab = (value*)data;
    value vector = tab[0];
    value nb_device = tab[1];
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int seek;
    int tag;
    bigArray = Field (Field(vector, 1), 0);
    seek = Int_val(Field(vector, 10));
    size = Int_val(Field(vector, 4))-seek;
    dev_vec_array = Field(vector, 2);
    dev_vec = Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;

    int custom = 0;
    GET_TYPE_SIZE;
    if (d_A) {
      cuMemFree((CUdeviceptr)(d_A));
    }
    print_gpu_free("GPU_FREE", Field(vector, 9), Int_val(nb_device), "CUDA", size*type_size);
    free(tab);
#else
    if (data) {
      cuMemFree((CUdeviceptr)(data));
    }
#endif
  }



  CAMLprim value spoc_cuda_device_to_cpu(value vector, value nb_device, value gi, 
                                         value device, value queue_id) {
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    CUdevice dev;
    void* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int id;
    int size;
    int type_size;
    int tag;
    int seek;

    cuda_event_list *evt;
    id = Int_val(Field(vector, 0));
    seek = Int_val(Field(vector, 10));

    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(vector, 2);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;
    size = Int_val(Field(vector, 4))-seek;

    CUDA_GET_CONTEXT;

    CUDA_CHECK_CALL(cuDeviceGet(&dev, Int_val(nb_device)));

  //h_A = (void*)Data_bigarray_val(bigArray);
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec);
    int custom = 0;
    GET_TYPE_SIZE;

#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;    
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
    value* data_table = malloc(3 * sizeof(value));
    data_table[0] = vector;
    data_table[1] = nb_device;
#endif

    CUDA_CHECK_CALL(cuMemcpyDtoHAsync((void*)h_A+seek*type_size, d_A+seek*type_size, 
                                      size*type_size, queue[Int_val(queue_id)]));

#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    
    int event_id = print_start_transfert("DEVICE_TO_CPU", size*type_size, 
                                         Int_val(Field(vector, 9)), "CUDA", 
                                         Int_val(Field(gi, 7)), NULL, events);
    data_table[2] = Val_int(event_id);
    CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], 
                                        cuda_free_after_transfer, (void*)data_table, 0));
#else	
    CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], 
                                        cuda_free_after_transfer, (void*)d_A, 0));
#endif
    /*
      CUDA_CHECK_CALL(cuEventCreate(&(evt->evt), CU_EVENT_BLOCKING_SYNC));
      evt->vec = d_A;
      evt->next = (cuda_event_list*)(Field(device,3));
      Store_field(device, 3 , (value) (evt));
      CUDA_CHECK_CALL(cuEventRecord(evt->evt, queue[Int_val(queue_id)]));
    */
    CUDA_RESTORE_CONTEXT;


    CAMLreturn(Val_unit);
  }


  CAMLprim value spoc_cuda_free_vect(value vector, value nb_device){
    CAMLparam2(vector, nb_device);
    CAMLlocal3(bigArray,  dev_vec_array, dev_vec);
    CUdevice dev;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int id;
    enum cudaError_enum cuda_error = 0;
    id = Int_val(Field(vector, 0));

    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(vector, 2);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;

    if (&d_A)
      CUDA_CHECK_CALL(cuMemFree(d_A));

    Store_field(dev_vec, 1, (value)NULL);

#ifdef PROFILE
    int type_size;
    int size;
    int seek;
    int tag;
    seek = Int_val(Field(vector, 10));
    size = Int_val(Field(vector, 4))-seek;
    int custom = 0;
    GET_TYPE_SIZE;
    print_gpu_free("GPU_FREE", Field(vector, 9), nb_device, "CUDA", size*type_size);
#endif

    CAMLreturn(Val_unit);
  }


  CAMLprim value spoc_cuda_alloc_vector(value vector, value nb_device, value gi, int custom){
    CAMLparam3(vector, nb_device, gi);
    CAMLlocal3(bigArray,  dev_vec_array, dev_vec);
    CUdevice dev = 0;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int type_size;
    int tag;
    int size;
    size_t size_left;
    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(vector, 2);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    if (cuv){
      if (cuv->cu_vector)
	cuMemFree(cuv->cu_vector);
      free(cuv);
    }
    cuv = (cu_vector*)malloc(sizeof(cu_vector));

    size = Int_val(Field(vector, 4));
    cuv->cu_size = size;

    CUDA_GET_CONTEXT;
    GET_TYPE_SIZE;


    CUDA_CHECK_CALL(cuMemAlloc(&cuv->cu_vector, size*type_size));


#ifdef PROFILE
    print_gpu_alloc("GPU_ALLOC", Int_val(Field(vector, 9)), nb_device, "CUDA", size*type_size);
#endif

    CUDA_RESTORE_CONTEXT;

    Store_field(dev_vec, 1, (value)cuv);

    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_alloc_vect(value vector, value nb_device, value gi){
    return spoc_cuda_alloc_vector(vector, nb_device, gi, 0);
  }

  CAMLprim value spoc_cuda_custom_alloc_vect(value vector, value nb_device, value gi){
    return spoc_cuda_alloc_vector(vector, nb_device, gi, 1);
  }


  CAMLprim value spoc_opencl_free_vect(value vector, value nb_device){
    CAMLparam2(vector, nb_device);
    CAMLlocal3(bigArray,dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));

    if (d_A) {
      clReleaseMemObject(d_A);
      Store_field(dev_vec,1,NULL);
    }

    Store_field(dev_vec, 1, Val_cl_mem(d_A));

#ifdef PROFILE
    int type_size;
    int size;
    int seek;
    int tag;
    seek = Int_val(Field(vector, 10));
    size = Int_val(Field(vector, 4))-seek;
    int custom = 0;
    //bigArray = Field (Field(vector, 1), 0);
    GET_TYPE_SIZE;
    print_gpu_free("GPU_FREE", Field(vector, 9), nb_device, "OPENCL", size*type_size);
#endif

    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_alloc_vect(value vector, value nb_device, value gi){
    CAMLparam3(vector, nb_device, gi);
    CAMLlocal3(bigArray, dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    cl_device_id device_id;
    int type_size;
    int tag;
    bigArray = Field (Field(vector, 1), 0);
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec);
  //h_A = (void*)Data_bigarray_val(bigArray);
    dev_vec_array = Field(vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));
    size =Int_val(Field(vector, 4));
    //if (d_A){ clReleaseMemObject(d_A);}
    OPENCL_GET_CONTEXT;
    int custom = 0;
    GET_TYPE_SIZE;
    
#ifdef PROFILE
    print_gpu_alloc("GPU_ALLOC", Int_val(Field(vector, 9)), nb_device, 
                    "OPENCL", size*type_size);
#endif

    OPENCL_CHECK_CALL1(d_A, clCreateBuffer(ctx, CL_MEM_READ_WRITE, size*type_size, NULL, 
                                           &opencl_error));
    OPENCL_RESTORE_CONTEXT;

    Store_field(dev_vec, 1, Val_cl_mem(d_A));
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_custom_alloc_vect(value vector, value nb_device, value gi){
    CAMLparam3(vector, nb_device, gi);
    CAMLlocal3(customArray, dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    cl_device_id device_id;
    int type_size;
    int tag;

    customArray = Field (Field(vector, 1), 0);

    type_size = Int_val(Field(Field(customArray, 1),0))*sizeof(char);

    size = Int_val(Field(vector, 4));

    dev_vec_array = Field(vector, 3);

    dev_vec =Field(dev_vec_array, Int_val(nb_device));

    d_A = Cl_mem_val(Field(dev_vec, 1));

    size = Int_val(Field(vector, 4));


    OPENCL_GET_CONTEXT;

    OPENCL_CHECK_CALL1(d_A, clCreateBuffer(ctx, CL_MEM_READ_WRITE, size*type_size, NULL, 
                                           &opencl_error));


#ifdef PROFILE
    print_gpu_alloc("GPU_ALLOC", Int_val(Field(vector, 9)), 
                    nb_device, "OPENCL", size*type_size);
#endif

    OPENCL_RESTORE_CONTEXT;

    Store_field(dev_vec, 1, Val_cl_mem(d_A));
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_part_device_to_cpu(value vector, value sub_vector, value nb_device,
                                              value gi, value gc_info, value queue_id, 
                                              value host_offset, value guest_offset, 
                                              value start, value part_size){
    CAMLparam5(vector, nb_device, gi, gc_info, queue_id);
    CAMLxparam5(sub_vector, host_offset, guest_offset, start, part_size);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int size;
    int type_size;
    int tag;
    spoc_cuda_gc_info* gcInfo;
    bigArray = Field (Field(vector, 1), 0);
    dev_vec_array = Field(sub_vector, 2);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;
    CUDA_GET_CONTEXT;
    int custom = 0;
    GET_TYPE_SIZE;

    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec)+
      ((Long_val(host_offset)+Long_val(start))*type_size);
  //    h_A = (char*)Data_bigarray_val(bigArray)+((Long_val(host_offset)+Long_val(start))*type_size);

    size = Int_val(Field(sub_vector, 4));

#ifdef PROFILE
    CUevent start_transfer;
    CUevent end_transfer;
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
    value* data_table = malloc(3 * sizeof(value));
    data_table[0] = vector;
    data_table[1] = nb_device;    
#endif

    CUDA_CHECK_CALL(cuMemcpyDtoHAsync(h_A+(Long_val(guest_offset)*type_size), d_A /* (gcInfo->curr_ptr)*/,  (Long_val(part_size))*type_size, queue[Int_val(queue_id)]));
    
#ifdef PROFILE
    cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
    cuda_events* events = malloc(sizeof(cuda_events));
    events->start = start_transfer;
    events->end = end_transfer;
    
    int event_id = print_start_part_transfert("PART_DEVICE_TO_CPU", 
                                              ((Long_val(part_size))*type_size), 
                                              (Int_val(Field(vector, 4))*type_size), 
                                              Int_val(Field(sub_vector, 9)), 
                                              Int_val(Field(vector, 9)), "CUDA", 
                                              Int_val(nb_device), NULL, events);;
    data_table[2] = Val_int(event_id);
    CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], cuda_free_after_transfer, 
                                        (void*)data_table, 0));
#else	
    CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], cuda_free_after_transfer,
                                        (void*)d_A, 0));
#endif
    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_part_device_to_cpu_b(value* tab_val, value nb_val){
    value vector = tab_val[0];
    value sub_vector = tab_val[1];
    value nb_device = tab_val[2];
    value gi = tab_val[3];
    value gc_info = tab_val[4];
    value queue_id = tab_val[5];
    value host_offset = tab_val[6];
    value guest_offset = tab_val[7];
    value start = tab_val[8];
    value part_size = tab_val[9];
    return spoc_cuda_part_device_to_cpu(vector, sub_vector, nb_device, gi, gc_info, 
                                        queue_id, host_offset, guest_offset,  start, part_size);
  }
  CAMLprim value spoc_cuda_part_device_to_cpu_n(value vector, value subvector, value nb_device, 
                                                value gi, value gc_info, value queue_id, 
                                                value host_offset, value guest_offset, value start,
                                                value part_size){
    return spoc_cuda_part_device_to_cpu(vector, subvector, nb_device, gi, gc_info, queue_id, 
                                        host_offset, guest_offset, start, part_size);
  }

  CAMLprim value spoc_cuda_custom_part_device_to_cpu_b()
  {
    return Val_unit;
  }
  CAMLprim value spoc_cuda_custom_part_device_to_cpu_n()
  {
    return Val_unit;
  }

  CAMLprim value spoc_opencl_custom_part_device_to_cpu_b()
  {
    return Val_unit;
  }
  CAMLprim value spoc_opencl_custom_part_device_to_cpu_n()
  {
    return Val_unit;
  }

  CAMLprim value spoc_cuda_custom_device_to_cpu(value vector, value nb_device, value gi, 
                                                value queue_id) {
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLlocal5(device_vec, vec, customArray,  dev_vec_array, dev_vec);
    CUdevice dev;
    void* h_A;
    cu_vector* cuv;
    CUdeviceptr d_A;
    int id;
    int size;
    int type_size;
    int tag;

    id = Int_val(Field(vector, 0));
    customArray = Field (Field(vector, 1), 0);
    h_A = (void*)Field(Field(customArray, 0), 1);
    type_size = Int_val(Field(Field(customArray, 1),0))*sizeof(char);
    size = Int_val(Field(vector, 4));
    dev_vec_array = Field(vector, 2);
    dev_vec = Field(dev_vec_array, Int_val(nb_device));
    cuv = (cu_vector*)Field(dev_vec, 1);
    d_A = cuv->cu_vector;

    CUDA_GET_CONTEXT;


    CUDA_CHECK_CALL(cuDeviceGet(&dev, Int_val(nb_device)));

#ifdef PROFILE
    value* data_table = malloc(3 * sizeof(value));
    data_table[0] = vector;
    data_table[1] = nb_device;
    CUevent start_transfer;
    CUevent end_transfer;    
    cuEventCreate(&start_transfer, CU_EVENT_DEFAULT);
    cuEventCreate(&end_transfer, CU_EVENT_DEFAULT);
    cuEventRecord(start_transfer, queue[Int_val(queue_id)]);
#endif

   CUDA_CHECK_CALL(cuMemcpyDtoHAsync((void*)h_A, d_A, size*type_size, queue[Int_val(queue_id)]));
    
#ifdef PROFILE
   cuEventRecord(end_transfer, queue[Int_val(queue_id)]);
   cuda_events* events = malloc(sizeof(cuda_events));
   events->start = start_transfer;
   events->end = end_transfer; 
   int event_id = print_start_transfert("DEVICE_TO_CPU_CUSTOM", size*type_size, 
                                        Int_val(Field(vector, 9)), "CUDA", 
                                        Int_val(Field(gi, 7)), NULL, events);
   
   data_table[2] = Val_int(event_id);
   CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], 
                                       cuda_free_after_transfer, (void*)data_table, 0));
#else	
   CUDA_CHECK_CALL(cuStreamAddCallback(queue[Int_val(queue_id)], 
                                       cuda_free_after_transfer, (void*)d_A, 0));
#endif
    
   CUDA_RESTORE_CONTEXT;
   
   CAMLreturn(Val_unit);
  }



  value spoc_opencl_cpu_to_device(value vector, value nb_device, value gi, value queue_id) {
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    cl_device_id device_id;
    int type_size;
    int tag;
    bigArray = Field (Field(vector, 1), 0);
  //h_A = (void*)Data_bigarray_val(bigArray);
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec);
    dev_vec_array = Field(vector, 3);

    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));
    size =Int_val(Field(vector, 4));
    int custom = 0;
    GET_TYPE_SIZE;
    OPENCL_GET_CONTEXT;

#ifdef PROFILE
    cl_event event_transfert;
    OPENCL_CHECK_CALL1(
      opencl_error, clEnqueueWriteBuffer(queue[Int_val(queue_id)], d_A, CL_FALSE,
                                         0, size*type_size, h_A, 0, NULL, &event_transfert));
    
    print_start_transfert("CPU_TO_DEVICE", 
                          size*type_size, Int_val(Field(vector, 9)), "OPENCL", 
                          Int_val(Field(gi, 7)), event_transfert, NULL);
#else
    OPENCL_CHECK_CALL1(
      opencl_error, clEnqueueWriteBuffer (queue[Int_val(queue_id)], d_A, CL_FALSE,
                                          0, size*type_size, h_A, 0, NULL, NULL));
#endif
    OPENCL_RESTORE_CONTEXT;

    Store_field(dev_vec, 1, Val_cl_mem(d_A));
    Store_field(dev_vec_array, Int_val(nb_device), dev_vec);
    CAMLreturn(Val_unit);
  }
  CAMLprim value spoc_opencl_custom_cpu_to_device(value vector, value nb_device, value gi, 
                                                  value queue_id) {
    CAMLparam4(vector, nb_device, gi, queue_id);
    CAMLlocal5(device_vec, vec, customArray,  dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    cl_device_id device_id;
    int type_size;
    int tag;
    int seek;

    customArray = Field (Field(vector, 1), 0);
    h_A = (void*)Field(Field(customArray, 0), 1);
    seek = Int_val(Field(vector, 10));

    type_size = Int_val(Field(Field(customArray, 1),0));
    size = Int_val(Field(vector, 4))-seek;

    dev_vec_array = Field(vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));
    OPENCL_GET_CONTEXT;

    OPENCL_TRY("clGetContextInfo", clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 
                                                    (size_t)sizeof(cl_device_id), 
                                                    &device_id, NULL));
#ifdef PROFILE
    cl_event event_transfert;
    OPENCL_CHECK_CALL1(opencl_error, 
                       clEnqueueWriteBuffer(queue[Int_val(queue_id)], d_A,
                                            CL_FALSE, seek, size*type_size, 
                                            h_A, seek, NULL, &event_transfert));

    print_start_transfert("CPU_TO_DEVICE_CUSTOM", size*type_size, 
                          Int_val(Field(vector, 9)), "OPENCL", Int_val(Field(gi, 7)), 
                          event_transfert, NULL);

#else
    OPENCL_CHECK_CALL1(opencl_error, clEnqueueWriteBuffer(queue[Int_val(queue_id)], d_A,
      CL_FALSE, seek, size*type_size, h_A, seek, NULL, NULL));
#endif
    OPENCL_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }





  CAMLprim value spoc_opencl_part_cpu_to_device(value vector, value sub_vector, value nb_device,
      value gi, value gc_info, value queue_id, value host_offset, value guest_offset, 
      value start, value part_size){
    CAMLparam5(vector, nb_device, gi, gc_info, queue_id);
    CAMLxparam5(sub_vector, host_offset, guest_offset, start, part_size);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    cl_device_id device_id;
    int type_size;
    int tag;
    bigArray = Field (Field(vector, 1), 0);
    int custom = 0;
    GET_TYPE_SIZE
      h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec)+
      //      h_A = (void*)((char*)Data_bigarray_val(bigArray)+
      ((Long_val(host_offset)+Long_val(start))*type_size);
  //);
    dev_vec_array = Field(sub_vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));
    size = Int_val(Field(sub_vector, 4));

    OPENCL_GET_CONTEXT;
 
#ifdef PROFILE
    cl_event event_transfert;
    
    OPENCL_CHECK_CALL1(opencl_error,
      clEnqueueWriteBuffer(queue[Int_val(queue_id)], d_A, CL_FALSE,
      (Long_val(guest_offset)*type_size),
      Long_val(part_size)*type_size,
      h_A, 0, NULL, &event_transfert));
    
    print_start_part_transfert("PART_CPU_TO_DEVICE", ((Long_val(part_size))*type_size), 
      (Int_val(Field(vector, 4))*type_size),
      Int_val(Field(sub_vector, 9)), Int_val(Field(vector, 9)), "OPENCL", Int_val(nb_device), 
      event_transfert, NULL);
#else
    OPENCL_CHECK_CALL1(opencl_error,
      clEnqueueWriteBuffer(queue[Int_val(queue_id)], d_A, CL_FALSE,
      (Long_val(guest_offset)*type_size),
      Long_val(part_size)*type_size,
      h_A, 0, NULL, NULL));
#endif

    OPENCL_RESTORE_CONTEXT;
    Store_field(dev_vec, 1, Val_cl_mem(d_A));
    Store_field(dev_vec_array, Int_val(nb_device), dev_vec);

    CAMLreturn(Val_unit);
  }


  CAMLprim value spoc_opencl_part_cpu_to_device_b(value* tab_val, value nb_val){
    value vector = tab_val[0];
    value sub_vector = tab_val[1];
    value nb_device = tab_val[2];
    value gi = tab_val[3];
    value gc_info = tab_val[4];
    value queue_id = tab_val[5];
    value host_offset = tab_val[6];
    value guest_offset = tab_val[7];
    value start = tab_val[8];
    value part_size = tab_val[9];
    return spoc_opencl_part_cpu_to_device(vector, sub_vector, nb_device, gi, gc_info, queue_id, host_offset, 0, 0, part_size);
  }
  CAMLprim value spoc_opencl_part_cpu_to_device_n(value vector, value subvector, value nb_device, value gi, value gc_info, value queue_id, value host_offset, value guest_offset, value start, value part_size){
    return spoc_opencl_part_cpu_to_device(vector, subvector, nb_device, gi, gc_info, queue_id, host_offset, guest_offset, start, part_size);
  }


  value spoc_opencl_device_to_device() {
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);

    CAMLreturn(Val_unit);

  }



  CAMLprim value spoc_opencl_device_to_cpu(value vector, value nb_device, value gi, value si, value queue_id) {
    CAMLparam5(vector, nb_device, gi, si, queue_id);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    int type_size;
    int tag;
    cl_device_id device_id;
    cl_command_queue  q;
    bigArray = Field (Field(vector, 1), 0);
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec);
  //h_A = (void*)Data_bigarray_val(bigArray);
    int custom = 0;
    GET_TYPE_SIZE

      dev_vec_array = Field(vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));

    size = Long_val(Field(vector, 4));
    OPENCL_GET_CONTEXT;

    q = queue[Int_val(queue_id)];

#ifdef PROFILE
    cl_event event_transfer;
 
    OPENCL_CHECK_CALL1(opencl_error, clEnqueueReadBuffer(q, d_A, CL_FALSE, 0, size*type_size,
      h_A, 0, NULL, &event_transfer));
    
    print_start_transfert("DEVICE_TO_CPU", size*type_size, Int_val(Field(vector, 9)), 
      "OPENCL", Int_val(Field(gi, 7)), event_transfer, NULL);
#else
    OPENCL_CHECK_CALL1(opencl_error, clEnqueueReadBuffer(q, d_A, CL_FALSE, 0, size*type_size,
      h_A, 0, NULL, NULL));
#endif
    clReleaseMemObject(d_A);
    Store_field(dev_vec,1,NULL);
    OPENCL_CHECK_CALL1(opencl_error, clFlush(queue[Int_val(queue_id)]));

    OPENCL_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_custom_device_to_cpu(value vector, value nb_device, value gi, value si, value queue_id) {
    CAMLparam5(vector, nb_device, gi, si, queue_id);
    CAMLlocal5(device_vec, vec, customArray,  dev_vec_array, dev_vec);
    void* h_A;
    cl_mem d_A;
    int size;
    int type_size;
    int tag;
    cl_device_id device_id;

    customArray = Field (Field(vector, 1), 0);

    h_A = (void*)Field(Field(customArray, 0), 1);

    dev_vec_array = (value)Field(vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));
    size =Long_val(Field(vector, 4));
    type_size = Long_val(Field(Field(customArray, 1),0))*sizeof(char);

    OPENCL_GET_CONTEXT;

    OPENCL_TRY("clGetContextInfo", clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 
                                                    (size_t)sizeof(cl_device_id), 
                                                    &device_id, NULL)) ;
#ifdef PROFILE
    cl_event event_transfert;
    OPENCL_CHECK_CALL1(opencl_error, clEnqueueReadBuffer(queue[Int_val(queue_id)], d_A, 
                                                         CL_FALSE, 0, size*type_size, h_A, 0, 
                                                         NULL, &event_transfert));
    print_start_transfert("DEVICE_TO_CPU_CUSTOM", size*type_size, 
                          Int_val(Field(vector, 9)), "OPENCL", Int_val(Field(gi, 7)), 
                          event_transfert, NULL);
#else
    OPENCL_CHECK_CALL1(opencl_error, clEnqueueReadBuffer(queue[Int_val(queue_id)], d_A, 
                                                         CL_FALSE, 0, size*type_size, h_A, 0, 
                                                         NULL, NULL));
#endif

    clReleaseMemObject(d_A);
    Store_field(dev_vec,1,NULL);

    OPENCL_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }





  CAMLprim value spoc_opencl_part_device_to_cpu(value vector, value sub_vector, value nb_device, value gi, value gc_info, value queue_id, value host_offset, value guest_offset, value start, value part_size){
    CAMLparam5(vector, nb_device, gi, gc_info, queue_id);
    CAMLxparam5(sub_vector, host_offset, guest_offset, start, part_size);
    CAMLlocal5(device_vec, vec, bigArray,  dev_vec_array, dev_vec);
    void* h_A;

    cl_mem d_A;
    int size;
    int type_size;
    int tag;
    cl_device_id device_id;
    cl_command_queue  q;

    bigArray = Field (Field(vector, 1), 0);
    int custom = 0;
    GET_TYPE_SIZE
    h_A =(void*)(((host_vector*)(Field(Field(bigArray,0),1)))->vec)+
      //      h_A = (void*)((char*)Data_bigarray_val(bigArray)+
      ((Long_val(host_offset)+Long_val(start))*type_size);
  //);

    dev_vec_array = Field(sub_vector, 3);
    dev_vec =Field(dev_vec_array, Int_val(nb_device));
    d_A = Cl_mem_val(Field(dev_vec, 1));

    size = Int_val(Field(sub_vector, 4));
    OPENCL_GET_CONTEXT;

    q = queue[Int_val(queue_id)];

#ifdef PROFILE
    
#endif

#ifdef PROFILE
    cl_event event_transfert;
    OPENCL_CHECK_CALL1(opencl_error,
                       clEnqueueReadBuffer(q, d_A, CL_FALSE,
                                           (Long_val(guest_offset)*type_size),
                                           (Long_val(part_size))*type_size,
                                           h_A, 0, NULL, &event_transfert));
    print_start_part_transfert("PART_DEVICE_TO_CPU", ((Long_val(part_size))*type_size),
                                  (Int_val(Field(vector, 4))*type_size),
                                  Int_val(Field(sub_vector, 9)), Int_val(Field(vector, 9)),
                                  "OPENCL", Int_val(nb_device), event_transfert, NULL);
#else
    OPENCL_CHECK_CALL1(opencl_error,
                       clEnqueueReadBuffer(q, d_A, CL_FALSE,
                                           (Long_val(guest_offset)*type_size),
                                           (Long_val(part_size))*type_size,
                                           h_A, 0, NULL, NULL));
#endif

    OPENCL_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_part_device_to_cpu_b(value* tab_val, value nb_val){
    value vector = tab_val[0];
    value sub_vector = tab_val[1];
    value nb_device = tab_val[2];
    value gi = tab_val[3];
    value gc_info = tab_val[4];
    value queue_id = tab_val[5];
    value host_offset = tab_val[6];
    value guest_offset = tab_val[7];
    value start = tab_val[8];
    value part_size = tab_val[9];
    return spoc_opencl_part_device_to_cpu(vector, sub_vector, nb_device, gi, gc_info, queue_id, host_offset, guest_offset, start, part_size);
  }
  CAMLprim value spoc_opencl_part_device_to_cpu_n(value vector, value subvector, value nb_device, value gi, value gc_info, value queue_id, value host_offset, value guest_offset, value start, value part_size){
    return spoc_opencl_part_device_to_cpu(vector, subvector, nb_device, gi, gc_info, queue_id, host_offset, guest_offset, start, part_size);
  }


  CAMLprim value spoc_opencl_vector_copy(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }
  CAMLprim value spoc_cuda_vector_copy(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    CAMLparam5(vectorA, startA, vectorB, startB, size);
    CAMLxparam2(gi, nb_device);
    CAMLlocal1(bigArray);
    CAMLlocal3(device_vecA, dev_vec_arrayA, dev_vecA);
    CAMLlocal3(device_vecB, dev_vec_arrayB, dev_vecB);
    //void* h_A;
    cu_vector* cuvA;
    cu_vector* cuvB;
    CUdeviceptr d_A;
    CUdeviceptr d_B;

    //	int size;
    int type_size;
    int tag;
    bigArray = Field (Field(vectorA, 1), 0);
    dev_vec_arrayA = Field(vectorA, 2);
    dev_vecA =Field(dev_vec_arrayA, Int_val(nb_device));
    cuvA = (cu_vector*)Field(dev_vecA, 1);
    d_A = cuvA->cu_vector;
    dev_vec_arrayB = Field(vectorB, 2);
    dev_vecB =Field(dev_vec_arrayB, Int_val(nb_device));
    cuvB = (cu_vector*)Field(dev_vecB, 1);
    d_B = cuvB->cu_vector;

    CUDA_GET_CONTEXT;

    int custom = 0;
    GET_TYPE_SIZE;

    CUDA_CHECK_CALL(cuMemcpy(d_A+(Int_val(startA)*type_size), d_B+(Int_val(startB)*type_size), size*type_size));

    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_vector_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value startA = tab_val[1];
    value vectorB = tab_val[2];
    value startB = tab_val[3];
    value size = tab_val[4];
    value gi = tab_val[5];
    value nb_device = tab_val[6];
    return spoc_opencl_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }
  CAMLprim value spoc_cuda_vector_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value startA = tab_val[1];
    value vectorB = tab_val[2];
    value startB = tab_val[3];
    value size = tab_val[4];
    value gi = tab_val[5];
    value nb_device = tab_val[6];
    return spoc_cuda_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }

  CAMLprim value spoc_opencl_vector_copy_n(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    return spoc_opencl_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }
  CAMLprim value spoc_cuda_vector_copy_n(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    return spoc_cuda_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }


  CAMLprim value spoc_opencl_custom_vector_copy(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_custom_vector_copy(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_custom_vector_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value startA = tab_val[1];
    value vectorB = tab_val[2];
    value startB = tab_val[3];
    value size = tab_val[4];
    value gi = tab_val[5];
    value nb_device = tab_val[6];
    return spoc_opencl_custom_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }
  CAMLprim value spoc_cuda_custom_vector_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value startA = tab_val[1];
    value vectorB = tab_val[2];
    value startB = tab_val[3];
    value size = tab_val[4];
    value gi = tab_val[5];
    value nb_device = tab_val[6];
    return spoc_cuda_custom_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }

  CAMLprim value spoc_opencl_custom_vector_copy_n(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    return spoc_opencl_custom_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }
  CAMLprim value spoc_cuda_custom_vector_copy_n(value vectorA, value startA, value vectorB, value startB, value size, value gi, value nb_device){
    return spoc_cuda_custom_vector_copy( vectorA, startA, vectorB, startB, size, gi, nb_device);
  }




  CAMLprim value spoc_opencl_matrix_copy(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_matrix_copy(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    CAMLparam5(vectorA, ld_A, start_rowA, start_colA, vectorB);
    CAMLxparam5(ld_B, start_rowB, start_colB, rows, cols);
    CAMLxparam2(gi, nb_device);
    CAMLlocal1(bigArray);
    CAMLlocal3(device_vecA, dev_vec_arrayA, dev_vecA);
    CAMLlocal3(device_vecB, dev_vec_arrayB, dev_vecB);
    //void* h_A;
    cu_vector* cuvA;
    CUdeviceptr d_A;
    cu_vector* cuvB;
    CUdeviceptr d_B;

    //	int size;
    int type_size;
    int tag;
    CUDA_MEMCPY2D pCopy ;

    bigArray = Field (Field(vectorA, 1), 0);
    dev_vec_arrayA = Field(vectorA, 2);
    dev_vecA =Field(dev_vec_arrayA, Int_val(nb_device));
    cuvA = (cu_vector*)Field(dev_vecA, 1);
    d_A = cuvA->cu_vector;

    dev_vec_arrayB = Field(vectorB, 2);
    dev_vecB =Field(dev_vec_arrayB, Int_val(nb_device));
    cuvB = (cu_vector*)Field(dev_vecB, 1);
    d_B = cuvB->cu_vector;

    CUDA_GET_CONTEXT;


    int custom = 0;
    GET_TYPE_SIZE;

    pCopy.srcXInBytes = Int_val(start_colA)*type_size;
    pCopy.srcY = Int_val(start_rowA);
    pCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.srcDevice = d_A;
    pCopy.srcPitch = Int_val(ld_A);

    pCopy.dstXInBytes = Int_val(start_colB)*type_size;
    pCopy.dstY = Int_val(start_rowB);
    pCopy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    pCopy.dstDevice = d_B;
    pCopy.dstPitch = Int_val(ld_B);

    pCopy.WidthInBytes = Int_val(cols)*type_size;
    pCopy.Height = Int_val(rows);

    printf (" pCopy : %ld %ld  %ld - %ld %ld %ld -- %ld %ld\n",
	    pCopy.srcXInBytes, pCopy.srcY, pCopy.srcPitch,
	    pCopy.dstXInBytes, pCopy.dstY, pCopy.dstPitch,
	    pCopy.WidthInBytes , pCopy.Height);
    fflush(stdout);

    d_A += pCopy.srcY * pCopy.srcPitch + pCopy.srcXInBytes;
    d_B += pCopy.dstY * pCopy.dstPitch + pCopy.dstXInBytes;
    CUDA_CHECK_CALL(cuMemcpy(d_B, d_A, (pCopy.WidthInBytes*pCopy.Height)));

    CUDA_RESTORE_CONTEXT;
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_matrix_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value ld_A = tab_val[1];
    value start_rowA = tab_val[2];
    value start_colA = tab_val[3];
    value vectorB = tab_val[4];
    value ld_B = tab_val[5];
    value start_rowB = tab_val[6];
    value start_colB = tab_val[7];
    value rows = tab_val[8];
    value cols = tab_val[9];
    value gi = tab_val[10];
    value nb_device = tab_val[11];
    return spoc_opencl_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }
  CAMLprim value spoc_cuda_matrix_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value ld_A = tab_val[1];
    value start_rowA = tab_val[2];
    value start_colA = tab_val[3];
    value vectorB = tab_val[4];
    value ld_B = tab_val[5];
    value start_rowB = tab_val[6];
    value start_colB = tab_val[7];
    value rows = tab_val[8];
    value cols = tab_val[9];
    value gi = tab_val[10];
    value nb_device = tab_val[11];
    return spoc_cuda_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }

  CAMLprim value spoc_opencl_matrix_copy_n(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    return spoc_opencl_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }
  CAMLprim value spoc_cuda_matrix_copy_n(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    return spoc_cuda_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }


  CAMLprim value spoc_opencl_custom_matrix_copy(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_cuda_custom_matrix_copy(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    CAMLparam0();
    UNIMPLENTED;
    fflush(stdout);
    CAMLreturn(Val_unit);
  }

  CAMLprim value spoc_opencl_custom_matrix_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value ld_A = tab_val[1];
    value start_rowA = tab_val[2];
    value start_colA = tab_val[3];
    value vectorB = tab_val[4];
    value ld_B = tab_val[5];
    value start_rowB = tab_val[6];
    value start_colB = tab_val[7];
    value rows = tab_val[8];
    value cols = tab_val[9];
    value gi = tab_val[10];
    value nb_device = tab_val[11];
    return spoc_opencl_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }
  CAMLprim value spoc_cuda_custom_matrix_copy_b(value* tab_val, value nb_val){
    value vectorA = tab_val[0];
    value ld_A = tab_val[1];
    value start_rowA = tab_val[2];
    value start_colA = tab_val[3];
    value vectorB = tab_val[4];
    value ld_B = tab_val[5];
    value start_rowB = tab_val[6];
    value start_colB = tab_val[7];
    value rows = tab_val[8];
    value cols = tab_val[9];
    value gi = tab_val[10];
    value nb_device = tab_val[11];
    return spoc_cuda_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }

  CAMLprim value spoc_opencl_custom_matrix_copy_n(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    return spoc_opencl_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }
  CAMLprim value spoc_cuda_custom_matrix_copy_n(value vectorA, value ld_A, value start_rowA, value start_colA, value vectorB, value ld_B, value start_rowB, value start_colB, value rows, value cols, value gi, value nb_device){
    return spoc_cuda_matrix_copy( vectorA, ld_A, start_rowA, start_colA, vectorB, ld_B, start_rowB, start_colB, rows, cols, gi, nb_device);
  }



  extern CAMLprim value caml_ba_set_1(value vb, value vind1, value newval);
  extern CAMLprim value caml_ba_get_1(value vb, value vind1);



  CAMLprim value spoc_set (value vect, value offset, value newval){
    CAMLparam3(vect, offset, newval);
    CAMLlocal1(bigarray);
    bigarray = Field(vect,0);
    caml_ba_set_1(bigarray, offset, newval);
    CAMLreturn(Val_unit);
  }


  CAMLprim value spoc_get (value vect, value offset){
    CAMLparam2(vect, offset);
    CAMLlocal2(bigarray, ret);
    bigarray = Field(vect,0);
    ret = caml_ba_get_1(bigarray, offset);
    CAMLreturn(ret);
  }

  CAMLprim value custom_getsizeofbool()
  {
    CAMLparam0();
    CAMLreturn(Val_int(sizeof(int)));
  }



  CAMLprim value custom_boolget (value customArray, value idx)
  {
    CAMLparam2(customArray, idx);
    int *b;
    b = ((int*)(Field(customArray, 1)))+(Int_val(idx));
    CAMLreturn(Val_bool(*b));
  }

  CAMLprim value custom_boolset (value customArray, value idx, value v)
  {
    CAMLparam3(customArray, idx, v);
    int* b;
    int i;
    i = Int_val(idx);
    b = ((int*)(Field(customArray, 1)))+i;

    *b = Bool_val(v);
    CAMLreturn(Val_unit);
  }

#ifdef __cplusplus
}
#endif
