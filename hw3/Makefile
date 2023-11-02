CFLAGS = -Drestrict=__restrict__ -O3 -g
MPICC = mpicxx
LDLIBS = -lm -lrt -lpthread

BIN =  program

.PHONY: all
all: $(BIN) 


program: program.cpp  utils.o
	$(MPICC) $(CFLAGS) -o program program.cpp  utils.o $(LDLIBS)
	
.PHONY:	clean
clean:
	rm -f $(BIN) 
