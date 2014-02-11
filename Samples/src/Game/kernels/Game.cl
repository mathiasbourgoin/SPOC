__kernel void mandelbrot(__global int* A, const int N, const int largeur, const int hauteur){
  int idx = get_global_id(0);

  int y = idx / hauteur;
  int x = idx - (y * largeur);

  if (y >= hauteur || x >= largeur)
  return;
  else
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


__kernel void game(__global int* A, const int N, const int largeur, const int hauteur){
	int idx = get_global_id(0);
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
    // barrier (CLK_GLOBAL_MEM_FENCE);
    if ((me == 1) && ((res < 2) || (res > 3)))
    		A[idx] = 0;
		else 
		if ((me == 0) && ((res == 3) ))
			A[idx] = 1;		
}
