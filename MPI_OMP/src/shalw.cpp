#include <stdlib.h>
#include <shalw.h>
#include <parse_args.hpp>
#include <memory.h>
#include <init.h>
#include <forward.h>
#include <export.h>
#include <mpi.h>
#include <math.h>   /* pour le rint */
#include <time.h>   /* chronometrage */
#include "sys/time.h"

double *hFil, *uFil, *vFil, *hPhy, *uPhy, *vPhy;
double *g_hFil;//, *g_uFil, *g_vFil, *g_hPhy, *g_uPhy, *g_vPhy;
int size_x, size_y;
int g_size_x, g_size_y, nb_steps;
double dx, dy, dt, pcor, grav, dissip, hmoy, alpha, height, epsilon;
bool file_export;
std::string export_path;

double my_gettimeofday(){
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

int main(int argc, char **argv) {
	/* Par processeur */ 
	int rang; // rang
	int NP; // NP = nombre de processus
	int provided;
	/* Initialisation MPI */
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	MPI_Comm_size(MPI_COMM_WORLD, &NP);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	/* Variables liees au chronometrage */
	double debut, fin;

	/* debut du chronometrage */
	debut = my_gettimeofday();    

	parse_args(argc, argv);
	printf("Command line options parsed\n");

	if (rang ==0){
		alloc();
		printf("Memory allocated by rank 0\n"); 

		/* Initialisations / calcul */		
		gauss_init();
		printf("State initialised\n");	
	}

	size_y = g_size_y ;
	size_x = (rang==0 || rang==NP-1)?(g_size_x/NP +1):(g_size_x/NP +2);

	loc_alloc();

	MPI_Scatter(&G_HFIL(0,0,0) /*sbuf*/, g_size_x/NP*g_size_y /*scount*/, MPI_DOUBLE /*sdtype*/, &HFIL(0,(rang!=0),0) /*rbuf*/, g_size_x/NP*g_size_y /*rcount*/, MPI_DOUBLE /*rdtype*/, 0 /*root*/, MPI_COMM_WORLD /*comm*/);


	forward(NP, rang);
	printf("State computed\n");

	if(rang==0){
		dealloc();
		printf("Global memory freed\n");
	}
	loc_dealloc();
	printf("Local memory freed. Rank = %d \n", rang);

	MPI_Finalize();
	/* fin du chronometrage */
	fin = my_gettimeofday();
	printf("Temps total de calcul : %g seconde(s) \n", fin - debut);

  return EXIT_SUCCESS;
}

