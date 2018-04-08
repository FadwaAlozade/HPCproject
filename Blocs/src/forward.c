#include <stdio.h>
#include <math.h>
#include <shalw.h>
#include <export.h>
#include <mpi.h>

double hFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //HPHY(t - 1, i, j) est encore nul
  if (t <= 2)
    return HPHY(t, i, j);
  return HPHY(t - 1, i, j) +
    alpha * (HFIL(t - 1, i, j) - 2 * HPHY(t - 1, i, j) + HPHY(t, i, j));
}

double uFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //UPHY(t - 1, i, j) est encore nul
  if (t <= 2)
    return UPHY(t, i, j);
  return UPHY(t - 1, i, j) +
    alpha * (UFIL(t - 1, i, j) - 2 * UPHY(t - 1, i, j) + UPHY(t, i, j));
}

double vFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //VPHY(t - 1, i, j) est encore nul
  if (t <= 2)
    return VPHY(t, i, j);
  return VPHY(t - 1, i, j) +
    alpha * (VFIL(t - 1, i, j) - 2 * VPHY(t - 1, i, j) + VPHY(t, i, j));
}

double hPhy_forward(int t, int i, int j) {
  double c, d;
  
  c = 0.;
  if (i > 0)
    c = UPHY(t - 1, i - 1, j);

  d = 0.;
  if (j < size_y - 1)
    d = VPHY(t - 1, i, j + 1);

  return HFIL(t - 1, i, j) -
    dt * hmoy * ((UPHY(t - 1, i, j) - c) / dx +
		 (d - VPHY(t - 1, i, j)) / dy);
}

double uPhy_forward(int t, int i, int j) {
  double b, e, f, g;
  
  if (i == size_x - 1)
    return 0.;

  b = 0.;
  if (i < size_x - 1)
    b = HPHY(t - 1, i + 1, j);

  e = 0.;
  if (j < size_y - 1)
    e = VPHY(t - 1, i, j + 1);

  f = 0.;
  if (i < size_x - 1)
    f = VPHY(t - 1, i + 1, j);

  g = 0.;
  if (i < size_x - 1 && j < size_y - 1)
    g = VPHY(t - 1, i + 1, j + 1);

  return UFIL(t - 1, i, j) +
    dt * ((-grav / dx) * (b - HPHY(t - 1, i, j)) +
	  (pcor / 4.) * (VPHY(t - 1, i, j) + e + f + g) -
	  (dissip * UFIL(t - 1, i, j)));
}

double vPhy_forward(int t, int i, int j) {
  double c, d, e, f;

  if (j == 0)
    return 0.;

  c = 0.;
  if (j > 0)
    c = HPHY(t - 1, i, j - 1);

  d = 0.;
  if (i > 0 && j > 0)
    d = UPHY(t - 1, i -1, j -1);

  e = 0.;
  if (i > 0)
    e = UPHY(t - 1, i - 1, j);

  f = 0.;
  if (j > 0)
    f = UPHY(t - 1, i, j - 1);

  return VFIL(t - 1, i, j) +
    dt * ((-grav / dy) * (HPHY(t - 1, i, j) - c) -
	  (pcor / 4.) * (d + e + f + UPHY(t - 1, i, j)) -
	  (dissip * VFIL(t - 1, i, j)));
}

void forward(int NP, int rang) {
	MPI_Status status;
	MPI_Datatype col_type;
	MPI_Datatype block, block_type;
	int TAG_FIRST_COL = 99;
	int TAG_LAST_COL = 100;
	int TAG_FIRST_ROW = 0;
	int TAG_LAST_ROW = 1;
	FILE *file = NULL;
	double svdt = 0.;
	int t = 0;
	int NBdim = sqrt(NP);

	MPI_Type_vector(g_size_x/NBdim, g_size_y/NBdim, g_size_y, MPI_DOUBLE,&block);
	MPI_Type_commit(&block);
	MPI_Type_create_resized(block, 0, g_size_y/NBdim*sizeof(double), &block_type);
 	MPI_Type_commit(&block_type);

	printf("Avant Scatter \n");
	MPI_Scatter(g_hFil /*sbuf*/, 1 /*scount*/, block_type /*sdtype*/, hFil+1*((rang%NBdim)!=0)+size_y*(!((rang>=0)&&(rang<NBdim))) /*rbuf*/, 1 /*rcount*/, block_type /*rdtype*/, 0 /*root*/, MPI_COMM_WORLD /*comm*/);

	printf("Après Scatter \n");

	if (rang==0) {
		if (file_export) {
			file = create_file();
			export_step(file, t);
		}
	}

	MPI_Type_vector(size_x, 1, size_y, MPI_DOUBLE,&col_type);
	MPI_Type_commit(&col_type);
	for (t = 1; t < nb_steps; t++) {
		/* Récupération et envoi des lignes à la frontière avec les proc voisins */
		// Envoi des colonnes
		printf("");
		if (rang%NBdim!=0)	MPI_Send(hFil+1, 1, col_type, rang-1, TAG_FIRST_COL, MPI_COMM_WORLD);
		printf("First column Sent. Rank = %d\n", rang);
		if (rang%NBdim!=NBdim-1)  MPI_Send(hFil+size_y-2, 1, col_type, rang+1, TAG_LAST_COL, MPI_COMM_WORLD);
		printf("Last column Sent. Rank = %d\n", rang);
		if (rang%NBdim!=NBdim-1)  MPI_Recv(hFil+size_y-1, 1, col_type, rang+1, TAG_FIRST_COL, MPI_COMM_WORLD, &status);
		printf("First column Received. Rank = %d\n", rang);
		if (rang%NBdim!=0)     MPI_Recv(hFil, 1, col_type, rang-1, TAG_LAST_COL, MPI_COMM_WORLD, &status);
		printf("Last column Received. Rank = %d\n", rang);

		// Envoi des lignes 
		if ( !((rang>=0)&&(rang<NBdim)))     MPI_Send(hFil+size_y, size_y, MPI_DOUBLE, rang-NBdim, TAG_FIRST_ROW, MPI_COMM_WORLD);
		printf("First row Sent. Rank = %d\n", rang);
		if ( !((rang>=NP-NBdim)&&(rang<NP)))  MPI_Send(hFil+size_y*(size_x-2), size_y, MPI_DOUBLE, rang+NBdim, TAG_LAST_ROW, MPI_COMM_WORLD);
		printf("Last row Sent. Rank = %d\n", rang);
		if ( !((rang>=NP-NBdim)&&(rang<NP)))  MPI_Recv(hFil+size_y*(size_x-1), size_y, MPI_DOUBLE, rang+NBdim, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
		printf("First row Received. Rank = %d\n", rang);
		if ( !((rang>=0)&&(rang<NBdim)))     MPI_Recv(hFil, size_y, MPI_DOUBLE, rang-NBdim, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
		printf("Last row Received. Rank = %d\n", rang);

		if (t == 1) {
			svdt = dt;
			dt = 0;
		}
		if (t == 2){
			dt = svdt / 2.;
		}

		for (int j = 0; j < size_y; j++) {
			for (int i = 0; i < size_x; i++) {
				HPHY(t, i, j) = hPhy_forward(t, i, j);
				UPHY(t, i, j) = uPhy_forward(t, i, j);
				VPHY(t, i, j) = vPhy_forward(t, i, j);
				HFIL(t, i, j) = hFil_forward(t, i, j);
				UFIL(t, i, j) = uFil_forward(t, i, j);
				VFIL(t, i, j) = vFil_forward(t, i, j);
			}
		}

		MPI_Gather(hFil+1*((rang%NBdim)!=0)+size_y*(!((rang>=0)&&(rang<NBdim))), 1 , block_type, g_hFil, 1, block_type, 0, MPI_COMM_WORLD);

		if (rang==0){
			if (file_export) {
				export_step(file, t);
			}
		}

		if (t == 2) {
			dt = svdt;
		}
	}

	if (rang==0){
		if (file_export) {
			finalize_export(file);
			//printf("\n\n");
		}
	}

}
