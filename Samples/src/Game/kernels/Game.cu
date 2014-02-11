#ifdef __cplusplus
extern "C" {
#endif
__global__ void mandelbrot(int* A, const int N, const int largeur, const int hauteur){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  int y = idx / hauteur;
  int x = idx - (y * largeur);

  if (y < hauteur && x < largeur)
    {
      int cpt = 0;
      float x1 = 0.;
      float y1 = 0.;
      


      float x2 = 0.;
      float y2 = 0.;
      float a = 4. * x / largeur - 2.;
      float b = 4. * y / hauteur - 2.;

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



__global__ void game(int* A, const int N, const int largeur, const int hauteur){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	  int y = idx / hauteur;
	  int x = idx - (y * largeur);
	 if (y >= hauteur || x >= largeur)
	  return;

	 int me = A[idx];
	 int north =  0 ;
	 int northEast = 0;
	 int northWest = 0;
	 int south = 0;
	 int southEast = 0;
	 int southWest = 0;
	 int east = 0;
	 int west = 0;
	 if (x > 0)
		 west = A[idx -1];
	 if (x < largeur - 1)
		 east = A[idx + 1];
	 if (y > 0)
		 north = A[idx - largeur];
     if (y < hauteur - 1)
		 south = A[idx + largeur];		  
	 
	 if ((y < hauteur - 1) && (x < largeur - 1))
		 southEast =  A[idx + largeur + 1];
	 if ((y < hauteur - 1) && (x > 0))
		 southWest =  A[idx + largeur - 1];
	 if ((y > 0) && (x >0))
		 northWest =  A[idx - largeur - 1];
	 if ((y > 0) && (x < largeur - 1))
		 northEast =  A[idx - largeur + 1];
	 int res = north + south + east + west + northEast + northWest + southEast + southWest;
	//__syncthreads();
    if ((me == 1) && (res < 2) || (res > 3))
    		A[idx] = 0;
		else 
		if ((me == 0) &&  (res == 3))
			A[idx] = 1;		
}
#ifdef __cplusplus
}
#endif
