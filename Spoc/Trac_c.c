#include "Trac_c.h"
#include <caml/threads.h>
#include <execinfo.h>
#define _GNU_SOURCE
#include <errno.h>

extern char* __progname;

struct timeval start, end;
int part_count = 0;
int size_sum = 0;
int info_elem = 0;
transfert_prof_info* info_tab = NULL;

void sync_event_prof(){
  int i;
  for(i = 0; i < info_elem; i++){
    cl_event event = info_tab[i].event;
    int event_id = info_tab[i].event_id;
    size_t size = info_tab[i].size;
    int vect_id = info_tab[i].vect_id;
    int part_id = info_tab[i].part_id;
    /* if(part_id == -1){ */
    /*   stop_transfert_callback("OPENCL_TRANSFER", event_id, size, vect_id, event); */
    /* } */
    /* else{ */
    /*   stop_part_transfert_callback("OPENCL_TRANSFER", event_id, size, part_id, event); */
    /* } */
  }
  info_elem = 0;
}

int start_transfert_callback(const char* desc, size_t size, int vect_id, const char* type_gpu, int id_device, cl_event event){
  gettimeofday(&start, NULL);
  printf("(%s) Memory transfer (%zu bytes) starting...\n", desc, size);
  static value* closure_f_start = NULL;
  if(closure_f_start == NULL){
    closure_f_start = caml_named_value("start_of_transfer");
  }
  value* args = malloc(6 * sizeof(value));
  args[0] = caml_copy_string(desc);
  args[1] = Val_int(size);
  args[2] = Val_int(vect_id);
  args[3] = caml_copy_string(type_gpu);
  args[4] = Val_int(id_device);
  args[5] = Val_bool(0);
  int res = (Int_val(caml_callbackN(*closure_f_start, 6, args)));
  if(event != NULL){
    info_elem++;
    info_tab = (transfert_prof_info *)realloc(info_tab, sizeof(transfert_prof_info) * info_elem);
    info_tab[info_elem - 1].event = event;
    info_tab[info_elem - 1].event_id = res;
    info_tab[info_elem - 1].size = size;
    info_tab[info_elem - 1].vect_id = vect_id;
    info_tab[info_elem - 1].part_id = -1;
  }

  // print backtrace to find caller
  int i,j, nptrs;
  #define SIZE 100
  void *buffer[SIZE];
  char **strings;
  nptrs = backtrace(buffer, SIZE);
  strings = backtrace_symbols(buffer, nptrs);
  for (i = 0; i < nptrs; i++){
    //    printf("%s\n", strings[i]);
    /*
 find first occurence of '(' or ' ' in message[i] and assume
     * everything before that is the file name. (Don't go beyond 0 though
     * (string terminator)*/
    size_t p = 0;
    while(strings[i][p] != '(' && strings[i][p] != ' '
	  && strings[i][p] != 0)
      ++p;

    char syscom[256];
    sprintf(syscom,"addr2line %p -e %.*s | grep `basename %s .asm`.ml ", buffer[i], p, strings[i], __progname);
    //last parameter is the file name of the symbol
    system(syscom);
  }

  fflush(stdout);
  free(strings);
  return res;
}

void stop_transfert_callback(const char* desc, int id, size_t size, int vect_id, cl_event event){
  printf("(%s) Memory transfer done (%zu bytes)| ", desc, size);
  gettimeofday(&end, NULL);
  double time_fct;
  if(event != NULL){
    clWaitForEvents(1 , &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    time_fct = (time_end - time_start) * 1.0e-3f; /*nano secondes -> micro secondes*/
  }
  else{
    long int time_elapsed = ((1000000 * end.tv_sec) + end.tv_usec) - ((1000000 * start.tv_sec) + start.tv_usec);
    //TEMP
    time_fct = (double)time_elapsed;
  }
  printf("Time elapsed : %f µs\n\n", time_fct);
  static value* closure_f_end = NULL;
  if(closure_f_end == NULL){
    closure_f_end = caml_named_value("end_of_transfer");
  }
  //caml_callback3(*closure_f_end, Val_int(vect_id), caml_copy_int64(time_elapsed),Val_int(id));
  caml_callback3(*closure_f_end, Val_int(vect_id), caml_copy_double(time_fct), Val_int(id));
}

int start_part_transfert_callback(const char* desc, size_t part_size, size_t total_size, int part_id, int vect_id, const char* type_gpu, int id_device, cl_event event){
  //size_sum += part_size;
  //part_count++;
  gettimeofday(&start, NULL);
  //printf("(%s) Memory transfer (%zu bytes) starting, part(%i)...\n", desc, part_size, part_count);
  static value* closure_f_part_start = NULL;
  if(closure_f_part_start == NULL){
    closure_f_part_start = caml_named_value("start_of_transfer_part");
  }
  value* args = malloc(8 * sizeof(value));
  args[0] = caml_copy_string(desc);
  args[1] = Val_int(part_size);
  args[2] = Val_int(total_size);
  args[3] = Val_int(part_id);
  args[4] = Val_int(vect_id);
  args[5] = caml_copy_string(type_gpu);
  args[6] = Val_int(id_device);
  args[7] = Val_bool(1);
  /*if(size_sum == total_size){
    printf("(%s) Last part transfer (%zu bytes in %i parts in total)\n", desc, total_size, part_count);
    part_count = 0;
    size_sum = 0;
    }*/
  int res = (Int_val(caml_callbackN(*closure_f_part_start, 6, args)));
  if(event != NULL){
    info_elem++;
    info_tab = realloc(info_tab, sizeof(transfert_prof_info) * info_elem);
    info_tab[info_elem - 1].event = event;
    info_tab[info_elem - 1].event_id = res;
    info_tab[info_elem - 1].size = part_size;
    info_tab[info_elem - 1].vect_id = vect_id;
    info_tab[info_elem - 1].part_id = part_id;
  }
  return res;
}

void stop_part_transfert_callback(const char* desc, int id, size_t size, int part_id, cl_event event){
  printf("(%s) Memory transfer done (%zu bytes)| ", desc, size);
  gettimeofday(&end, NULL);
  double time_fct;
  if(event != NULL){
    clWaitForEvents(1 , &event);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    time_fct = (time_end - time_start) * 1.0e-3f; /*nano secondes -> micro secondes*/
  }
  else{
    long int time_elapsed = ((1000000 * end.tv_sec) + end.tv_usec) - ((1000000 * start.tv_sec) + start.tv_usec);
    time_fct = (double)time_elapsed;
  }
  printf("Time elapsed : %f µs\n\n", time_fct);
  static value* closure_f_part_end = NULL;
  if(closure_f_part_end == NULL){
    closure_f_part_end = caml_named_value("end_of_transfer_part");
  }
  //	caml_callback3(*closure_f_part_end, Val_int(part_id), caml_copy_int64(time_elapsed), Val_int(id));
  caml_callback3(*closure_f_part_end, Val_int(part_id), caml_copy_double(time_fct), Val_int(id));
}

void gpu_free_callback(const char* desc, int vect_id, int id_device, const char* type_gpu, size_t size){
  printf("(%s)Vector n°%i freed on %s device n°%i (%zu bytes)\n", desc, vect_id, type_gpu, id_device, size);
  static value* closure_f_gpu_free = NULL;
  if(closure_f_gpu_free == NULL){
    closure_f_gpu_free = caml_named_value("gpu_free");
  }
  value* args = malloc(4 * sizeof(value));
  args[0] = caml_copy_string(desc);
  args[1] = Val_int(size);
  args[2] = Val_int(vect_id);
  args[3] = caml_copy_string(type_gpu);
  args[4] = Val_int(id_device);
  caml_callbackN(*closure_f_gpu_free, 5, args);
}

void gpu_alloc_callback(const char* desc, int vect_id, int id_device, const char* type_gpu, size_t size){
  printf("(%s)Vector n°%i allocated on %s device n°%i (%zu bytes)\n", desc, vect_id, type_gpu, id_device, size);
  static value* closure_f_gpu_alloc = NULL;
  if(closure_f_gpu_alloc == NULL){
    closure_f_gpu_alloc = caml_named_value("gpu_alloc");
  }
  value* args = malloc(4 * sizeof(value));
  args[0] = caml_copy_string(desc);
  args[1] = Val_int(size);
  args[2] = Val_int(vect_id);
  args[3] = caml_copy_string(type_gpu);
  args[4] = Val_int(id_device);
  caml_callbackN(*closure_f_gpu_alloc, 5, args);
}

void start_gpu_compile_callback(){
  gettimeofday(&start, NULL);
}

void stop_gpu_compile_callback(const char* desc, int id_device){
  gettimeofday(&end, NULL);
  long int time_elapsed = ((1000000 * end.tv_sec) + end.tv_usec) - ((1000000 * start.tv_sec) + start.tv_usec);
  static value* closure_f_gpu_compile = NULL;
  if(closure_f_gpu_compile == NULL){
    closure_f_gpu_compile = caml_named_value("gpu_compile");
  }
  caml_callback3(*closure_f_gpu_compile, caml_copy_string(desc),  caml_copy_int64(time_elapsed), Val_int(id_device));
}

int start_gpu_execution_callback(const char* desc, int id_device){
  static value* closure_f_gpu_exec_start = NULL;
  if(closure_f_gpu_exec_start == NULL){
    closure_f_gpu_exec_start = caml_named_value("start_of_exec");
  }
  int r = caml_callback2(*closure_f_gpu_exec_start, caml_copy_string(desc), Val_int(id_device));
  return Int_val(r);
}

void stop_gpu_execution_callback(int id, double duration){
  static value* closure_f_gpu_exec_stop = NULL;
  if(closure_f_gpu_exec_stop == NULL){
    closure_f_gpu_exec_stop = caml_named_value("end_of_exec");
  }
  caml_callback2(*closure_f_gpu_exec_stop, Val_int(id), caml_copy_double(duration));
}
