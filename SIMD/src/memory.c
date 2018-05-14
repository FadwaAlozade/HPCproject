#include <stdlib.h>
#include <shalw.h>

void alloc(void) {
  void *ptrhFil = NULL;
  posix_memalign((void**)&hFil, 32, 2*size_x*size_y*sizeof(double));
  void *ptruFil = NULL;
  posix_memalign((void**)&uFil, 32, 2*size_x*size_y*sizeof(double));
  void *ptrvFil = NULL;
  posix_memalign((void**)&vFil, 32, 2*size_x*size_y*sizeof(double));
  void *ptrhPhy = NULL;
  posix_memalign((void**)&hPhy, 32, 2*size_x*size_y*sizeof(double));
  void *ptruPhy = NULL;
  posix_memalign((void**)&uPhy, 32, 2*size_x*size_y*sizeof(double));
  void *ptrvPhy = NULL;
  posix_memalign((void**)&vPhy, 32, 2*size_x*size_y*sizeof(double));

}

void dealloc(void) {
  free(hFil);
  free(uFil);
  free(vFil);
  free(hPhy);
  free(uPhy);
  free(vPhy);
}
