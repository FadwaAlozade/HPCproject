#include <stdio.h>
#include <shalw.h>
#include <mpi.h>

MPI_File create_file(void) {
	MPI_File fh;
	char fname[256];

	sprintf(fname, "%s/shalw_%dx%d_T%d.sav", export_path.c_str(), g_size_x, g_size_y, nb_steps);

	MPI_File_open (MPI_COMM_WORLD, fname, (MPI_MODE_RDWR | MPI_MODE_CREATE), MPI_INFO_NULL, &fh);

	return fh;
}

void export_step(MPI_File fh, int t, int rang, int NP) {
	int TAILLEBLOC = g_size_x*g_size_y/NP;
	MPI_Offset offset = (g_size_y*g_size_x*t + rang*TAILLEBLOC) * sizeof(double); // Va à la position rang*g_size_x*g_size_y/NP (rang*tailledubloc) de l'image courante

	MPI_File_write_at(fh, offset, &HFIL(t,(rang!=0), 0), TAILLEBLOC, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void finalize_export(MPI_File *fh) {
  MPI_File_close(fh);
}

