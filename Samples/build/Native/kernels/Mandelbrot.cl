__kernel void mandelbrot(__global int* A, const int N, const int largeur, const int hauteur){
  int y = get_local_id (1) + (get_group_id(1) * get_local_size (1));
  int x = get_local_id (0) + (get_group_id(0) * get_local_size (0));	
  
  //blockDim.fx * blockIdx.fx + threadIdx.fx;
  
  //  return;
  
  if (y >= hauteur || x >= largeur)
    return
      ;
  {
    int cpt = 0;
    float x1 = 0.f;
    float y1 = 0.f;
    
    
    
    float x2 = 0.f;
    float y2 = 0.f;
    float a = 4.f * x / largeur - 2.f;
    float b = 4.f * y / hauteur - 2.f;
    
    float val = x1* x1 + y1 * y1;      
    
    while (cpt < N && val <= 4.f)
      {
	cpt ++;	  
	x2 = x1* x1 - y1 * y1 + a;
	y2 = 2.f * x1 * y1 + b;
	x1 = x2;
	y1 = y2;
	val = x1* x1 + y1 * y1;
      }
    
    A[y*hauteur+x] = cpt;
    
  }
}
