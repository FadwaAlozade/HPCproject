all:
	make -C Bandes
	make -C Blocs
	make -C Sequentiel
	make -C Recouvrement
	make -C MPI-IO

clean:
	make clean -C Bandes
	make clean -C Blocs
	make clean -C Sequentiel
	make clean -C Recouvrement
	make clean -C MPI-IO 
