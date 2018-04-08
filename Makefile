all:
	make -C Bandes
	make -C Blocs
	make -C Sequentiel

clean:
	make clean -C Bandes
	make clean -C Blocs
	make clean -C Sequentiel 
