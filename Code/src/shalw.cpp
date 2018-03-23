#include <stdlib.h>
#include <shalw.h>
#include <parse_args.hpp>
#include <memory.h>
#include <init.h>
#include <forward.h>
#include <export.h>
#include <mpi.h>

double *hFil, *uFil, *vFil, *hPhy, *uPhy, *vPhy;
double *g_hFil, *uFil_loc, *vFil_loc, *hPhy_loc, *uPhy_loc, *vPhy_loc;
int size_x, size_y, nb_steps;
int size_xp, size_yp, nbp_steps;
double dx, dy, dt, pcor, grav, dissip, hmoy, alpha, height, epsilon;
bool file_export;
std::string export_path;

int main(int argc, char **argv) {
 	/* Par processeur */ 
  int rang; // rang
  int NP; // NP = nombre de processus

  /* Initialisation MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &NP);
  MPI_Comm_rank(MPI_COMM_WORLD, &rang);

  if (rang ==0){
  	  parse_args(argc, argv);
  	  printf("Command line options parsed\n");
	  alloc();
	  printf("Memory allocated\n");  	
  }


  size_y = size_y / NP;
  size_x = size_x;
  
  loc_alloc();
  printf("Local memory allocated. Rang = %d \n", rang);  	

  
  gauss_init();
  printf("State initialised\n");

  forward();
  printf("State computed\n");
  
  dealloc();
  printf("Memory freed\n");

  MPI_Finalize();
  
  return EXIT_SUCCESS;
}
