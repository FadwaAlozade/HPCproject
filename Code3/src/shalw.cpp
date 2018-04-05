/* Note : nombre de blocs = nombre de processeurs !!  */
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
//int div; // nombre de blocs par lequel l'image sera divisée (4 ou 16)

double my_gettimeofday(){
	struct timeval tmp_time;
	gettimeofday(&tmp_time, NULL);
	return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

int main(int argc, char **argv) {
		/* Par processeur */ 
	int rang; // rang
	int NP; // NP = nombre de processus

	/* Initialisation MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NP);
	MPI_Comm_rank(MPI_COMM_WORLD, &rang);
	/* Variables liees au chronometrage */
	double debut, fin;

	if (NP!=4 && NP!=16){
		printf("Le nombre de processeurs choisi doit etre 4 ou bien 16. \n");
		return EXIT_FAILURE;
	}

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
        for (int j=0; j<size_y ; j++){
            printf(" hfil[%d] = %f\n", j, *(hFil+j));
        }
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    NBdim = sqrt(NP); // nombre de blocs par dimension (ligne ou colonne)
	size_y = (rang%NBdim==0 || rang%NBdim==NBdim-1)?(g_size_y/NBdim +1):(g_size_y/NBdim +2); // Si c'est les blocs de la première colonne ou la dernière colonne, on n'alloue que 1 colonne en plus
	size_x = ( ((rang>=0)&&(rang<NBdim-1)) || ((rang>=NP-NBdim)&&(rang<NP)) )?(g_size_x/NBdim +1):(g_size_x/NBdim +2); // Si c'est les blocs de la première ligne ( 0<=rang<NP-NBdim) ou dernière ligne ( NP-NBdim <=rang<NP), alors on n'alloue qu'une seule ligne en plus

	loc_alloc();
	printf("Local memory allocated. Rang = %d \n", rang); 
   

	/* Récupération et envoi des lignes à la frontière avec les proc voisins */
	// if (p!=0)     MPI_Send(local_im+w, w, MPI_UNSIGNED_CHAR, p-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
	// if (p!=NP-1)  MPI_Send(local_im+w*(local_h-2), w, MPI_UNSIGNED_CHAR, p+1, TAG_LAST_ROW, MPI_COMM_WORLD);
	// if (p!=NP-1)  MPI_Recv(local_im+w*(local_h-1), w, MPI_UNSIGNED_CHAR, p+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
	// if (p!=0)     MPI_Recv(local_im, w, MPI_UNSIGNED_CHAR, p-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);

	printf("Avant Scatter \n");
	MPI_Scatter(g_hFil /*sbuf*/, size_x/NP*size_y /*scount*/, MPI_DOUBLE /*sdtype*/, hFil+size_y*(rang!=0) /*rbuf*/, size_x/NP*size_y /*rcount*/, MPI_DOUBLE /*rdtype*/, 0 /*root*/, MPI_COMM_WORLD /*comm*/);
	printf("Après Scatter \n");

	forward(NP, rang);
	printf("State computed\n");

	dealloc();
	printf("Memory freed\n");

	MPI_Finalize();
	/* fin du chronometrage */
	fin = my_gettimeofday();
	printf("Temps total de calcul : %g seconde(s) \n", fin - debut);
  
  return EXIT_SUCCESS;
}
