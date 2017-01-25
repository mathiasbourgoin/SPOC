#ifndef _TRAC_C_H
#define _TRAC_C_H

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <math.h>

#include "Opencl_dynlink.h"
#include "Spoc.h"

typedef struct transfert_prof_info {
  cl_event event;
  int event_id;
  size_t size;
  int vect_id;
  int part_id;
} transfert_prof_info;

void sync_event_prof();
int start_transfert_callback(const char* desc, size_t size, int vect_id, const char* type_gpu, int id_device, cl_event event);
void stop_transfert_callback(const char* desc, int id, size_t size, int vect_id, cl_event event);
int start_part_transfert_callback(const char* desc, size_t part_size, size_t total_size, int part_id, int vect_id, const char* type_gpu, int id_device, cl_event event);
void stop_part_transfert_callback(const char* desc, int id, size_t size, int part_id, cl_event event);
void gpu_free_callback(const char* desc, int vect_id, int id_device, const char* type_gpu, size_t size);
void gpu_alloc_callback(const char* desc, int vect_id, int id_device, const char* type_gpu, size_t size);
void start_gpu_compile_callback();
void stop_gpu_compile_callback(const char* desc, int id_device);
int start_gpu_execution_callback(const char* desc, int id_device);
void stop_gpu_execution_callback(int id, double duration);
#endif
