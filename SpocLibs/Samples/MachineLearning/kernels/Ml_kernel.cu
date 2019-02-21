#ifdef __cplusplus
extern "C" {
#endif

__global__ void kernel_compute(int* trainingSet, int* data, int* res, int setSize, int dataSize){
  int diff, toAdd, computeId;
  computeId = blockIdx.x * blockDim.x + threadIdx.x;
  //__shared__ int set[784];
  if(computeId < setSize){
    diff = 0;
    for(int i = 0; i < dataSize; i++){
      toAdd = data[i] - trainingSet[computeId*784 + i];
      diff += toAdd * toAdd;
    }
    res[computeId] = diff;
  }
}

#ifdef __cplusplus
}
#endif
