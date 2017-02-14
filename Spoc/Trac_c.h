#ifndef _TRAC_C_H
#define _TRAC_C_H

#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <execinfo.h>

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <caml/threads.h>

#include "Opencl_dynlink.h"
#include "Spoc.h"

#define _GNU_SOURCE

#define GETTIME(time) do {                              \
    if (clock_gettime(CLOCK_MONOTONIC, &time) == -1) {  \
      fprintf(stdout, "Error getting time: %s\n",       \
              strerror(errno));                         \
    }                                                   \
  } while (0) 

#define INITIAL_TAB_SIZE 8

extern char* __progname;

typedef struct cuda_events {
  CUevent start;
  CUevent end;
} cuda_events;

typedef struct transfert_prof_info {
  struct timespec start;
  cl_event event_cl;
  cuda_events* event_cu;
  int event_id;
  size_t size;
  int vect_id;
  int part_id;
} transfert_prof_info;

CAMLprim value open_output_profiling();
CAMLprim value close_output_profiling();
CAMLprim value print_vector_bytecode(value* argv, int argn);
CAMLprim value print_vector_native(value v_id, value v_dev, value v_length, value v_size, 
                                   value v_kind, value v_is_sub, value v_depth, value v_start, 
                                   value v_ok_range, value v_ko_range, value v_parent_id);
CAMLprim value print_info_bytecode(value* argv, int argn);
CAMLprim value print_info_native(value v_name, value v_global_mem, value v_local_mem,
                                 value v_clock_rate, value v_total_const_mem, 
                                 value v_multi_proc_count, value v_ecc, value v_id,
                                 value v_spec_info, value v_print_comma);
CAMLprim value pre_print_device(value number);
CAMLprim value print_event(value desc);
CAMLprim value begin_event(value desc);
CAMLprim value end_event(value desc, value id);
void open_output_file();
void close_output_file();
double get_time();
int get_id_event();
void sync_event_prof();
int print_start_transfert(const char* desc, size_t size, int vect_id, const char* type_gpu,
                          int device_id, cl_event event_cl, cuda_events* event_cu);
void print_stop_transfert(const char* desc, int id, size_t size, int vect_id, 
                          cl_event event_cl, cuda_events* event_cu, struct timespec start);
int print_start_part_transfert(const char* desc, size_t part_size, size_t total_size, 
                               int part_id, int vect_id, const char* type_gpu, 
                               int device_id, cl_event event, cuda_events* event_cu);
void print_stop_part_transfert(const char* desc, int id, size_t size, int part_id, 
                               cl_event event_cl, cuda_events* event_cu, struct timespec start);
void print_gpu_free(const char* desc, int vect_id, int device_id, 
                    const char* type_gpu, size_t size);
void print_gpu_alloc(const char* desc, int vect_id, int device_id, 
                     const char* type_gpu, size_t size);
struct timespec* print_start_gpu_compile();
void print_stop_gpu_compile(const char* desc, int device_id, struct timespec* start);
int print_start_gpu_execution(const char* desc, int device_id);
void stop_gpu_execution_callback(int event_id, double duration);
void print_stop_gpu_execution(int event_id, double duration);
#endif
