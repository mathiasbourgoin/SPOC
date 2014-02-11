#ifdef __cplusplus
extern "C" {
#endif
__global__ void mandelbrot(int* A, const int N, const int largeur, const int hauteur, const int start_hauteur, const int end_hauteur){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int y = idx / hauteur;
  int x = idx - (y * largeur);

  if (y < (end_hauteur-start_hauteur) && x < largeur)
    {
      int cpt = 0;
      float x1 = 0.;
      float y1 = 0.;
      


      float x2 = 0.;
      float y2 = 0.;
      float a = 4. * x / largeur - 2.;
      float b = 4. * (y+start_hauteur) / hauteur - 2.;

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
