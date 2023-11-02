#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "const.h"
#include "types.h"
CLOG_EVAL mylog;

int main(int argc, char* argv[] ){
    int  flag = atoi(argv[1]);
    char command[256];
    int  ret_sys, np = atoi(argv[2]);
    //while ((np<<1)<=atoi(argv[2])) np = (np<<1);
    
    sprintf(command, "array_size = 2^%s * 2^%s\n", argv[3], argv[4]);
    if (flag) ret_sys = mylog.set_cmp(CMP_TO_SER, command);
    else      ret_sys = mylog.set_cmp(NO_CMP, command);
    if (ret_sys == FAIL) return EXIT_SUCCESS;    
    
    if (mylog.set_prog(GEN_PROG)==FAIL) return EXIT_SUCCESS;
    //sprintf(command, "%s %d %s %s %s", MPIRUN, np, GEN_DATA, argv[3], argv[4]);
    sprintf(command, "%s %d %s %s %s >> %s", MPIRUN, np, GEN_DATA, argv[3], argv[4], prog_stdout);
    ret_sys = system(command);
    if (mylog.time_end()==FAIL) {
    	printf("failed to generate input data\n");
    	return EXIT_SUCCESS;
    } 
    if ( flag ) {
    	//execute the serial version
    	if (mylog.set_prog(SER_PROG)==FAIL) return EXIT_SUCCESS;
    	//sprintf(command, "%s %s %s", SER_IMPL, argv[3], argv[4]);
		sprintf(command, "%s %s %s >> %s", SER_IMPL, argv[3], argv[4], prog_stdout);
		ret_sys = system(command);
		if (mylog.time_end()==FAIL) {
			printf("failed to generate serial reference output data\n");
			return EXIT_SUCCESS;
		}
		 
    	//execute the parallel reference version
    	if (mylog.set_prog(REF_PROG)==FAIL) return EXIT_SUCCESS; 
    	//sprintf(command, "%s %d %s %s %s", MPIRUN, np, REF_IMPL, argv[3], argv[4]);
		sprintf(command, "%s %d %s %s %s >> %s", MPIRUN, np, REF_IMPL, argv[3], argv[4], prog_stdout);
		ret_sys = system(command);
		if (mylog.time_end()==FAIL) {
			printf("failed to generate parallel reference output data\n");
			return EXIT_SUCCESS;
		}/**/ 
    } 
    //execute user's program
    if (mylog.set_prog(USER_PROG)==FAIL) return EXIT_SUCCESS;
    sprintf(command, "%s %d %s %s %s >> %s", MPIRUN, np,  USER_IMPL, argv[3], argv[4], prog_stdout);
    ret_sys = system(command);
	mylog.time_end()==FAIL;//if (mylog.time_end()==FAIL) return EXIT_SUCCESS;
    
    mylog.eval_perf()==FAIL;//if (mylog.eval_perf()==FAIL) return EXIT_SUCCESS;/**/ 

    return EXIT_SUCCESS;
}


