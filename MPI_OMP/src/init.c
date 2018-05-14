#include <math.h>
#include <shalw.h>
#include <stdlib.h>
#include <stdio.h>

void gauss_init(void) {
  double gmx, gmy, gsx, gsy;

  gmx = g_size_x * dx / 2 ;
  gmy = g_size_y * dy / 2 ;
  gsx = 25000 ;
  gsy = 25000 ;
  
  #pragma omp parallel for schedule(static) 
  for (int i = 0; i < g_size_x;  i++) {
    for (int j = 0; j < g_size_y; j++) {
      G_HFIL(0, i, j) = height *
	(exp(- pow((i * dx - gmx) / gsx, 2) / 2.)) *
	(exp(- pow((j * dy - gmy) / gsy, 2) / 2.)) ;
    
    }
  }
}
