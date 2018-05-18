#include <stdlib.h>
#include <stdio.h>
#include <shalw.h>



void alloc(void) {
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

void dealloc(void) {
  free(hFil);
  free(uFil);
  free(vFil);
  free(hPhy);
  free(uPhy);
  free(vPhy);
}
