#include "Trac_c.h"

struct timespec start_time;

int part_count = 0;
int size_sum = 0;
int event_counter = 0;
int info_elem = 0;
int info_tab_size = INITIAL_TAB_SIZE;

transfert_prof_info* info_tab = NULL;
FILE* output_file = NULL;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

CAMLprim value open_output_profiling(){
  CAMLparam0();
  open_output_file();
  CAMLreturn(Val_unit);
}

CAMLprim value close_output_profiling(){
  CAMLparam0();
  close_output_file();
  CAMLreturn(Val_unit);
}

CAMLprim value print_vector_bytecode(value* argv, int argn){
  return print_vector_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                           argv[6], argv[7], argv[8], argv[9], argv[10]);
}

CAMLprim value print_vector_native(value v_id, value v_dev, value v_length, value v_size, 
                            value v_kind, value v_is_sub, value v_depth, value v_start, 
                            value v_ok_range, value v_ko_range, value v_parent_id){
  CAMLparam5(v_id, v_dev, v_length, v_size, v_kind);
  CAMLxparam5(v_is_sub, v_depth, v_start, v_ok_range, v_ko_range);
  CAMLxparam1(v_parent_id);
  int id = Int_val(v_id);
  int dev = Int_val(v_dev);
  int length = Int_val(v_length);
  int size = Int_val(v_size);
  char* kind = String_val(v_kind);
  int is_sub = Bool_val(v_is_sub);
  int depth = Int_val(v_depth);
  int start = Int_val(v_start);
  int ok_range = Int_val(v_ok_range);
  int ko_range = Int_val(v_ko_range);
  int parent_id = Int_val(v_parent_id);
  pthread_mutex_lock(&mutex);
  fprintf(output_file, "{"
          "\"type\" : \"vector\",\n"
          "\"VectorId\" : %i,\n"
          "\"resides\" : %i,\n"
          "\"length\" : %i,\n"
          "\"size\" : %i, \n"
          "\"kind\" : \"%s\",\n"
          "\"isSub\" : %i", id, dev, length, (size*length), kind, is_sub);
  if(is_sub){
    fprintf(output_file, ",\n\"subVector\":{\n"
            "\"depth\" : %i,\n"
            "\"start\" : %i,\n"
            "\"okRange\" : %i,\n"
            "\"koRange\" : %i,\n"
            "\"parentID\" : %i\n"
            "}\n", depth, start, ok_range, ko_range, parent_id);
  }
  fprintf(output_file, "},\n");
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_unit);
}


CAMLprim value print_info_bytecode(value* argv, int argn){
  return print_info_native(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5],
                           argv[6], argv[7], argv[8], argv[9]);
}

CAMLprim value print_info_native(value v_name, value v_global_mem, value v_local_mem,
                          value v_clock_rate, value v_total_const_mem, 
                          value v_multi_proc_count, value v_ecc, value v_id,
                          value v_spec_info, value v_print_comma){
  CAMLparam5(v_name, v_global_mem, v_local_mem, v_clock_rate, v_total_const_mem);
  CAMLxparam5(v_multi_proc_count, v_ecc, v_id, v_spec_info, v_print_comma);
  char* name = String_val(v_name);
  int global_mem = Int_val(v_global_mem);
  int local_mem = Int_val(v_local_mem);
  int clock_rate = Int_val(v_clock_rate);
  int total_const_mem = Int_val(v_total_const_mem);
  int multi_proc_count = Int_val(v_multi_proc_count);
  int ecc = Bool_val(v_ecc);
  int id = Int_val(v_id);
  int print_comma = Bool_val(v_print_comma);
  char* spec_info = String_val(v_spec_info);
  pthread_mutex_lock(&mutex);
  fprintf(output_file, "{\n"
          "\"type\":\"device\","
          "\"generalInfo\": {"
          "\"type\":\"generalInfo\",\n"
          "\"name\":\"%s\",\n"
          "\"totalGlobalMem\":%i,\n"
          "\"localMemSize\":%i,\n"
          "\"clockRate\":%i,\n"
          "\"totalConstMem\":%i,\n"
          "\"multiProcessorCount\":%i,\n"
          "\"eccEnabled\":\"%i\",\n"
          "\"id\":%i\n},\n"
          "\"specificInfo\":\"%s\"\n"
          "}", name, global_mem, local_mem, clock_rate, total_const_mem, 
          multi_proc_count, ecc, id, spec_info);
  if(print_comma){
    fprintf(output_file, ",\n");
  }
  else{
    fprintf(output_file, "]},\n");
  }
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_unit);
}

CAMLprim value pre_print_device(value number){
  CAMLparam1(number);
  int nb = Int_val(number);
  pthread_mutex_lock(&mutex);
  fprintf(output_file, "{"
          "\"type\":\"deviceList\","
          "\"size\":%i,"
          "\"elem\":[", nb);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_unit);
}

CAMLprim value print_event(value desc){
  CAMLparam1(desc);
  char* description = String_val(desc);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"event\",\n"
          "\"desc\":\"%s\",\n"
          "\"id\":\"%i\",\n"
          "\"time\":\"%.3f\"\n"
          "},\n", description, event_id, time);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_unit);
}

CAMLprim value begin_event(value desc){
  CAMLparam1(desc);
  char* description = String_val(desc);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"%s\",\n"
          "\"etat\":\"start\",\n"
          "\"id\":\"%i\",\n"
          "\"startTime\":\"%.3f\"\n"
          "},\n", description, event_id, time);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_int(event_id));
}

CAMLprim value end_event(value desc, value id){
  CAMLparam2(desc, id);
  char* description = String_val(desc);
  int event_id = Int_val(id);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  fprintf(output_file, "{\n"
          "\"type\":\"%s\",\n"
          "\"etat\":\"end\",\n"
          "\"id\":\"%i\",\n"
          "\"endTime\":\"%.3f\"\n"
          "},\n", description, event_id, time);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  CAMLreturn(Val_unit);
}

void open_output_file(){
  if(output_file == NULL){
    GETTIME(start_time);
    info_tab = malloc(sizeof(transfert_prof_info) * INITIAL_TAB_SIZE);
    output_file = fopen("profilingInfo.json", "w");
    if(output_file == NULL){
      fprintf(stdout, "Error opening file: %s\n", strerror(errno));
    }
  }
  fprintf(output_file, "[\n");
}

void close_output_file(){
  sync_event_prof();
  fprintf(output_file, "\n {}]");
  fflush(output_file);
  if(output_file != NULL){
    fclose(output_file);
  }
  pthread_mutex_destroy(&mutex);
  free(info_tab);
}

double get_time(){
  struct timespec now;
  GETTIME(now);
  long int time_elapsed = (((1e9 * now.tv_sec) + now.tv_nsec) 
    - ((1e9 * start_time.tv_sec) + start_time.tv_nsec)) * 1.0e-3f; /*nano -> micro*/
  double time_fct = (double)time_elapsed;
  return time_fct;
}

int get_id_event(){
  ++event_counter;
  return event_counter;
}

void sync_event_prof(){
  int i;
  pthread_mutex_lock(&mutex);
  for(i = 0; i < info_elem; i++){
    int event_id = info_tab[i].event_id;
    size_t size = info_tab[i].size;
    int vect_id = info_tab[i].vect_id;
    int part_id = info_tab[i].part_id;
    struct timespec start = info_tab[i].start;
    if(info_tab[i].event_cl != NULL){
      if(part_id == -1){
        print_stop_transfert("OPENCL_TRANSFER", event_id, size, vect_id, 
                             info_tab[i].event_cl, NULL, start); 
      }
      else{
        print_stop_part_transfert("OPENCL_TRANSFER", event_id, size, part_id, 
                                  info_tab[i].event_cl, NULL, start); 
      }
    }
    if(info_tab[i].event_cu != NULL){
      if(part_id == -1){
        print_stop_transfert("CUDA_TRANSFER", event_id, size, vect_id, NULL,
                             info_tab[i].event_cu, start);
      }
      else{
        print_stop_part_transfert("OPENCL_TRANSFER", event_id, size, part_id, NULL, 
                                  info_tab[i].event_cu, start); 
      }
    }
  }
  info_elem = 0;
  pthread_mutex_unlock(&mutex);
}

int print_start_transfert(const char* desc, size_t size, int vect_id, const char* type_gpu,
                             int device_id, cl_event event_cl, cuda_events* event_cu){
  
  printf("(%s) Memory transfer (%zu bytes) starting...\n", desc, size);
  fflush(stdout);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int id_event = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"transfert\",\n"
          "\"desc\":\"%s\",\n"
          "\"state\":\"start\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"vectorId\":\"%i\",\n"
          "\"vectorSize\":\"%zu\",\n"
          "\"gpuType\":\"%s\",\n"
          "\"deviceId\":\"%i\",\n"
          "\"isSub\":\"false\"\n"
          "},\n", desc, time, id_event, vect_id, size, type_gpu, device_id);
  fflush(output_file);

  if(event_cl != NULL || event_cu != NULL){
    info_elem++;
    if(info_elem > info_tab_size){
      info_tab_size += 8;
      info_tab = realloc(info_tab, sizeof(transfert_prof_info) * info_tab_size); 
    }
    GETTIME(info_tab[info_elem - 1].start);
    if(event_cl == NULL){
      info_tab[info_elem - 1].event_cu = event_cu;
      info_tab[info_elem - 1].event_cl = NULL;
    }
    else{
      info_tab[info_elem - 1].event_cu = NULL;
      info_tab[info_elem - 1].event_cl = event_cl;
    }
    info_tab[info_elem - 1].event_id = id_event;
    info_tab[info_elem - 1].size = size;
    info_tab[info_elem - 1].vect_id = vect_id;
    info_tab[info_elem - 1].part_id = -1;
  }
  pthread_mutex_unlock(&mutex);
  return id_event;
  /*
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
   /* size_t p = 0;
    while(strings[i][p] != '(' && strings[i][p] != ' '
	  && strings[i][p] != 0)
      ++p;

    char syscom[256];
    sprintf(syscom,"addr2line %p -e %.*s | grep -v Spoc | grep -v stdlib | grep .ml ", buffer[i], p, strings[i], __progname);
    //last parameter is the file name of the symbol
    system(syscom);
  }

  fflush(stdout);
  free(strings);
  return res;*/
}

void print_stop_transfert(const char* desc, int id, size_t size, int vect_id, 
                             cl_event event_cl, cuda_events* event_cu ,struct timespec start){
  printf("(%s) Memory transfer done (%zu bytes)| ", desc, size);
  fflush(stdout);
  struct timespec end;
  GETTIME(end);
  double time_fct;
  if(event_cl != NULL){
    pthread_mutex_unlock(&mutex);
    clWaitForEvents(1 , &event_cl);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event_cl, CL_PROFILING_COMMAND_START, sizeof(time_start), 
                            &time_start, NULL);
    clGetEventProfilingInfo(event_cl, CL_PROFILING_COMMAND_END, sizeof(time_end), 
                            &time_end, NULL);
    time_fct = (time_end - time_start) * 1.0e-3f; /*nano secondes -> micro secondes*/
    pthread_mutex_lock(&mutex);
  }
  else{
    CUevent start = event_cu->start;
    CUevent end = event_cu->end;
    pthread_mutex_unlock(&mutex);
    cuEventSynchronize(start);
    float* duration = malloc(sizeof(float));
	cuEventElapsedTime(duration, start, end);
	time_fct = (double)((*duration) * 1000.0);
	cuEventDestroy(start);
	cuEventDestroy(end);
    free(duration);
    free(event_cu);
    pthread_mutex_lock(&mutex);
  }
  long int time_elapsed =  (((1e9 * end.tv_sec) + end.tv_nsec)
                            - ((1e9 * start.tv_sec) + start.tv_nsec)) * 1.0e-3f;
  printf("Time elapsed : %f µs\n\n", time_fct);
  fflush(stdout);
  double time = get_time();
  fprintf(output_file, "{\n"
          "\"type\":\"transfert\",\n"
          "\"id\":\"%i\",\n"
          "\"state\":\"end\",\n"
          "\"time\":\"%.3f\",\n"
          "\"vectorId\":\"%i\",\n"
          "\"duration\":\"%.3f\"\n"
          "},\n", id, time, vect_id, time_fct);
  fflush(output_file);
}

int print_start_part_transfert(const char* desc, size_t part_size, size_t total_size, 
                                  int part_id, int vect_id, const char* type_gpu, 
                                  int device_id, cl_event event_cl, cuda_events* event_cu){
  printf("(%s) Memory transfer (%zu bytes) starting\n", desc, part_size);
  fflush(stdout);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int id_event = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"part_transfert\",\n"
          "\"desc\":\"%s\",\n"
          "\"state\":\"start\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"partId\":\"%i\",\n"
          "\"vectorId\":\"%i\",\n"
          "\"partSize\":\"%zu\",\n"
          "\"vectorSize\":\"%zu\",\n"
          "\"gpuType\":\"%s\",\n"
          "\"deviceId\":\"%i\",\n"
          "\"isSub\":\"true\"\n"
          "},\n", desc, time, id_event, part_id, vect_id, part_size, total_size, 
          type_gpu, device_id);
  fflush(output_file);
  
  if(event_cl != NULL || event_cu != NULL){
    info_elem++;
    if(info_elem > info_tab_size){
      info_tab_size += 8;
      info_tab = realloc(info_tab, sizeof(transfert_prof_info) * info_tab_size); 
    }
    GETTIME(info_tab[info_elem - 1].start);
    if(event_cl == NULL){
      info_tab[info_elem - 1].event_cu = event_cu;
      info_tab[info_elem - 1].event_cl = NULL;
    }
    else{
      info_tab[info_elem - 1].event_cu = NULL;
      info_tab[info_elem - 1].event_cl = event_cl;
    }
    info_tab[info_elem - 1].event_id = id_event;
    info_tab[info_elem - 1].size = part_size;
    info_tab[info_elem - 1].vect_id = vect_id;
    info_tab[info_elem - 1].part_id = part_id;
  }
  pthread_mutex_unlock(&mutex);
  return id_event;
}

void print_stop_part_transfert(const char* desc, int id, size_t size, int part_id, 
                                  cl_event event_cl, cuda_events* event_cu, struct timespec start){
  printf("(%s) Memory transfer done (%zu bytes)| ", desc, size);
  fflush(stdout);
  struct timespec end;
  GETTIME(end);
  double time_fct;
  if(event_cl != NULL){
    pthread_mutex_unlock(&mutex);
    clWaitForEvents(1 , &event_cl);
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event_cl, CL_PROFILING_COMMAND_START, sizeof(time_start), 
                            &time_start, NULL);
    clGetEventProfilingInfo(event_cl, CL_PROFILING_COMMAND_END, sizeof(time_end), 
                            &time_end, NULL);
    time_fct = (time_end - time_start) * 1.0e-3f; /*nano secondes -> micro secondes*/
    pthread_mutex_lock(&mutex);
  }
  else{
    CUevent start = event_cu->start;
    CUevent end = event_cu->end;
    pthread_mutex_unlock(&mutex);
    cuEventSynchronize(start);
    float* duration = malloc(sizeof(float));
	cuEventElapsedTime(duration, start, end);
	time_fct = (double)((*duration) * 1000.0);
	cuEventDestroy(start);
	cuEventDestroy(end);
    free(duration);
    free(event_cu);
    pthread_mutex_lock(&mutex);
  }
  long int time_elapsed = (((1e9 * end.tv_sec) + end.tv_nsec) 
                           - ((1e9 * start.tv_sec) + start.tv_nsec)) * 1.0e-3f;
  printf("Time elapsed : %f µs\n\n", time_fct);
  fflush(stdout);
  double time = get_time();
  fprintf(output_file, "{\n"
          "\"type\":\"part_transfert\",\n"
          "\"id\":\"%i\",\n"
          "\"state\":\"end\",\n"
          "\"time\":\"%.3f\",\n"
          "\"partId\":\"%i\",\n"
          "\"duration\":\"%.3f\"\n"
          "},\n", id, time, part_id, time_fct);
  fflush(output_file);
}

void print_gpu_free(const char* desc, int vect_id, int device_id, 
                    const char* type_gpu, size_t size){
  printf("(%s)Vector n°%i freed on %s device n°%i (%zu bytes)\n", desc, vect_id, 
         type_gpu, device_id, size);
  fflush(stdout);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"freeGPU\",\n"
          "\"desc\":\"%s\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"vectorId\":\"%i\",\n"
          "\"vectorSize\":\"%zu\",\n"
          "\"gpuType\":\"%s\",\n"
          "\"deviceId\":\"%i\"\n"
          "},\n", desc, time, event_id, vect_id, size, type_gpu, device_id);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
}

void print_gpu_alloc(const char* desc, int vect_id, int device_id, 
                        const char* type_gpu, size_t size){
  printf("(%s)Vector n°%i allocated on %s device n°%i (%zu bytes)\n", desc, vect_id, 
         type_gpu, device_id, size);
  fflush(stdout);
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"allocGPU\",\n"
          "\"desc\":\"%s\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"vectorId\":\"%i\",\n"
          "\"vectorSize\":\"%zu\",\n"
          "\"gpuType\":\"%s\",\n"
          "\"deviceId\":\"%i\"\n"
          "},\n", desc, time, event_id, vect_id, size, type_gpu, device_id);

  fflush(output_file);
  pthread_mutex_unlock(&mutex);
}
struct timespec* print_start_gpu_compile(){
  struct timespec* start = malloc(sizeof(struct timespec));
  GETTIME(*start);
  return start;
}

void print_stop_gpu_compile(const char* desc, int device_id, struct timespec* start){
  struct timespec end;
  GETTIME(end);
  long int time_elapsed = (((1e9 * end.tv_sec) + end.tv_nsec)
                           - ((1e9 * start->tv_sec) + start->tv_nsec)) * 1.0e-3f;
  double time_fct = (double)time_elapsed; 
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"compile\",\n"
          "\"desc\":\"%s\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"duration\":\"%.3f\",\n"
          "\"deviceId\":\"%i\"\n"
          "},\n", desc, time, event_id, time_fct, device_id);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  free(start);
}

int print_start_gpu_execution(const char* desc, int device_id){
  double time = get_time();
  pthread_mutex_lock(&mutex);
  int event_id = get_id_event();
  fprintf(output_file, "{\n"
          "\"type\":\"execution\",\n"
          "\"desc\":\"%s\",\n"
          "\"state\":\"start\",\n"
          "\"time\":\"%.3f\",\n"
          "\"id\":\"%i\",\n"
          "\"deviceId\":\"%i\"\n"
          "},\n", desc, time, event_id, device_id);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
  return event_id;
}

void print_stop_gpu_execution(int event_id, double duration){
  double time = get_time();
  pthread_mutex_lock(&mutex);
  fprintf(output_file, "{\n"
          "\"type\":\"execution\",\n"
          "\"state\":\"end\",\n"
          "\"id\":\"%i\",\n"
          "\"time\":\"%.3f\",\n"
          "\"duration\":\"%.3f\"\n"
          "},\n", event_id, time, duration);
  fflush(output_file);
  pthread_mutex_unlock(&mutex);
}
