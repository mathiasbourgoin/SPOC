#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void kernel_compute(__global const int* trainingSet, __global const int* data, __global int* res, int setSize, int dataSize) {
  int diff, toAdd, computeId;
  computeId = get_global_id(0);
  if(computeId < setSize){
    diff = 0;
    for(int i = 0; i < dataSize; i++){
      toAdd = data[i] - trainingSet[computeId*dataSize + i];
      diff += toAdd * toAdd;
    }
    res[computeId] = diff;
  }
}
