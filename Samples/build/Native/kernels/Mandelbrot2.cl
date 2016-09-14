__kernel void mandelbrot(__global int* A, const int N, const int largeur, const int hauteur, const int start_hauteur, const int end_hauteur){
  int idx = get_global_id(0);

  int y = idx / hauteur;
  int x = idx - (y * largeur);

  if (y < (end_hauteur-start_hauteur) && x < largeur)
    {
      int cpt = 0;
      float x1 = 0.f;
      float y1 = 0.f;
      


      float x2 = 0.f;
      float y2 = 0.f;
      float a = 4.f * x / largeur - 2.f;
      float b = 4.f * (y+start_hauteur) / hauteur - 2.f;

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
