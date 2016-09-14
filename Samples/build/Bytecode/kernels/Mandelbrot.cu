#ifdef __cplusplus
extern "C" {
#endif
__global__ void mandelbrot(int* A, const int N, const int largeur, const int hauteur){

  int y = threadIdx.y + (blockIdx.y * blockDim.y);;
  int x = threadIdx.x + (blockIdx.x * blockDim.x);;

  if (y < hauteur && x < largeur)
    {
      int cpt = 0;
      float x1 = 0.f;
      float y1 = 0.f;
      


      float x2 = 0.f;
      float y2 = 0.f;
      float a = 4.f * x / largeur - 2.f;
      float b = 4.f * y / hauteur - 2.f;

      float val = x1* x1 + y1 * y1;      

      while (cpt < N && val <= 4.)
	{
	  cpt ++;	  
	  x2 = x1* x1 - y1 * y1 + a;
	  y2 = 2. * x1 * y1 + b;
	  x1 = x2;
	  y1 = y2;
	  val = x1* x1 + y1 * y1;
	}

      A[y*hauteur+x] = cpt;

    }
}
#ifdef __cplusplus
}
#endif
