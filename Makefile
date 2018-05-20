all:
	make -C Bandes
	make -C Blocs
	make -C Sequentiel
	make -C Recouvrement
	make -C MPI-IO
	make -C MPI_OMP
	make -C SIMD_seq
	make -C SIMD_MPI
	make -C SIMD_MPIOMP
	make -C SIMD_MPI-IO

clean:
	make clean -C Bandes
	make clean -C Blocs
	make clean -C Sequentiel
	make clean -C Recouvrement
	make clean -C MPI-IO 
	make clean -C MPI_OMP
	make clean -C SIMD_seq
	make clean -C SIMD_MPI
	make clean -C SIMD_MPIOMP
	make clean -C SIMD_MPI-IO

