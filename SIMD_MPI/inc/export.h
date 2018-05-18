#include <stdio.h>
#include <mpi.h>

MPI_File create_file(void);
void export_step(MPI_File, int, int, int);
void finalize_export(MPI_File*);
