CFLAGS = -Drestrict=__restrict__ -O3 -g
MPICC = mpicxx
LDLIBS = -lm -lrt -lpthread

OBJ = utils.o
BIN = program

.PHONY: all
all: $(BIN) $(OBJ)

program: program.cpp  utils.o
	$(MPICC) $(CFLAGS) -o program program.cpp  utils.o $(LDLIBS)
	
.PHONY:	clean
clean:
	rm -f $(BIN)
