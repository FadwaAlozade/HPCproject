#include <stdlib.h>
#include <stdio.h>
#include <shalw.h>

void alloc(void) {
  g_hFil = (double *) calloc(2*g_size_x*g_size_y, sizeof(double)); // on utilise deux grilles. En fonction de t, on accède soit à la première, soit à la deuxième.
}

void loc_alloc(void) {
  int a, b, c, d, e, f;
  a = posix_memalign((void**)&hFil, 32, 2*size_x*size_y*sizeof(double));
  b = posix_memalign((void**)&uFil, 32, 2*size_x*size_y*sizeof(double));
  c = posix_memalign((void**)&vFil, 32, 2*size_x*size_y*sizeof(double));
  d = posix_memalign((void**)&hPhy, 32, 2*size_x*size_y*sizeof(double));
  e = posix_memalign((void**)&uPhy, 32, 2*size_x*size_y*sizeof(double));
  f = posix_memalign((void**)&vPhy, 32, 2*size_x*size_y*sizeof(double));
  if (a || b || c || d || e || f){
    printf("Error in memory allocation\n");
  }
}

void loc_dealloc(void) {
  free(hFil);
  free(uFil);
  free(vFil);
  free(hPhy);
  free(uPhy);
  free(vPhy);

}


void dealloc(void){
	free(g_hFil);
}