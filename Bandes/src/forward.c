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
  int TAG_FIRST_ROW_UFIL = 0, TAG_FIRST_ROW_VFIL=1, TAG_FIRST_ROW_HFIL = 2, TAG_FIRST_ROW_UPHY =3 , TAG_FIRST_ROW_VPHY = 4, TAG_FIRST_ROW_HPHY = 5;
  int TAG_LAST_ROW_UFIL = 100, TAG_LAST_ROW_VFIL=101, TAG_LAST_ROW_HFIL = 102, TAG_LAST_ROW_UPHY = 103 , TAG_LAST_ROW_VPHY = 104, TAG_LAST_ROW_HPHY = 105;
  FILE *file = NULL;
  double svdt = 0.;
  int t = 0;
  
  if (rang==0) {
  	if (file_export) {
	    file = create_file();
	    export_step(file, t);
	  }
  }
  
  
  for (t = 1; t < nb_steps; t++) {
    /* Récupération et envoi des lignes à la frontière avec les proc voisins */

    // if (rang!=0)     MPI_Send(&HFIL(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
    // // //printf("First row Sent. Rank = %d\n", rang);
    // if (rang!=NP-1)  MPI_Send(&HFIL(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
    // //  // printf("Last row Sent. Rank = %d\n", rang);
    // if (rang!=NP-1)  MPI_Recv(&HFIL(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
    // //  // printf("First row Received. Rank = %d\n", rang);
    // if (rang!=0)     MPI_Recv(&HFIL(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
    // // //printf("Last row Received. Rank = %d\n", rang);
  	if (rang!=0) {
  		MPI_Sendrecv( &HFIL(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_HFIL/*sendtag*/, &HFIL(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_HFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  	}
  	if (rang!=NP-1) {
  		MPI_Sendrecv( &HFIL(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_HFIL/*sendtag*/, &HFIL(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_HFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  	}

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

    


   //MPI_Gather(hFil+size_y*(rang!=0), g_size_x/NP*g_size_y, MPI_DOUBLE, g_hFil, g_size_x/NP*g_size_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&HFIL(t,(rang!=0), 0),(g_size_x/NP)*g_size_y, MPI_DOUBLE, &G_HFIL(t, 0, 0), (g_size_x/NP)*g_size_y, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rang==0){
    	if (file_export) {
	      export_step(file, t);
	    }
    }
     
    if (t == 2) {
      dt = svdt;
    }

   //  if (rang!=0)    {
   //  	MPI_Send(&HPHY(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&UFIL(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&UPHY(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&VFIL(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&VPHY(t,1,0), size_y, MPI_DOUBLE, rang-1, TAG_FIRST_ROW, MPI_COMM_WORLD);
   //  } 
   //  if (rang!=NP-1)  {
   //  	MPI_Send(&HPHY(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&UFIL(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&UPHY(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&VFIL(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
   //  	MPI_Send(&VPHY(t,size_x-2,0) , size_y, MPI_DOUBLE, rang+1, TAG_LAST_ROW, MPI_COMM_WORLD);
   //  }
   //  if (rang!=NP-1)  {
   //  	MPI_Recv(&HPHY(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
   //  	MPI_Recv(&UFIL(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
   //  	MPI_Recv(&UPHY(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
   //  	MPI_Recv(&VFIL(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
   //  	MPI_Recv(&VPHY(t,size_x-1,0), size_y, MPI_DOUBLE, rang+1, TAG_FIRST_ROW, MPI_COMM_WORLD, &status);
   //  }
   //  if (rang!=0)    {
   //  	 MPI_Recv(&HPHY(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
   //  	 MPI_Recv(&UFIL(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
   //  	 MPI_Recv(&UPHY(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
   //  	 MPI_Recv(&VFIL(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
   //  	 MPI_Recv(&VPHY(t,0,0), size_y, MPI_DOUBLE, rang-1, TAG_LAST_ROW, MPI_COMM_WORLD, &status);
 		// }

    if (rang!=0) {
  		MPI_Sendrecv( &UFIL(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_UFIL/*sendtag*/, &UFIL(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_UFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &VFIL(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_VFIL/*sendtag*/, &VFIL(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_VFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &HPHY(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_HPHY/*sendtag*/, &HPHY(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_HPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &UPHY(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_UPHY/*sendtag*/, &UPHY(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_UPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &VPHY(t,1,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang-1/*dest*/, TAG_FIRST_ROW_VPHY/*sendtag*/, &VPHY(t,0,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang-1/*source*/, TAG_LAST_ROW_VPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  	}
  	if (rang!=NP-1) {
  		MPI_Sendrecv( &UFIL(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_UFIL/*sendtag*/, &UFIL(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_UFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &VFIL(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_VFIL/*sendtag*/, &VFIL(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_VFIL/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &HPHY(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_HPHY/*sendtag*/, &HPHY(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_HPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &UPHY(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_UPHY/*sendtag*/, &UPHY(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_UPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  		MPI_Sendrecv( &VPHY(t,size_x-2,0)/*sendbuf*/, size_y/*sendcount*/, MPI_DOUBLE/*sendtype*/, rang+1/*dest*/, TAG_LAST_ROW_VPHY/*sendtag*/, &VPHY(t,size_x-1,0)/*recvbuf*/, size_y/*recvcount*/, MPI_DOUBLE/*recvtype*/, rang+1/*source*/, TAG_FIRST_ROW_VPHY/*recvtag*/, MPI_COMM_WORLD/*comm*/, &status/*&status*/);
  	}

  }

  if (rang==0){
  	if (file_export) {
	    finalize_export(file);
	  }
  }
  
}
