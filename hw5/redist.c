#define _GNU_SOURCE
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h> 
#include <time.h>
#include <unistd.h>

#include <mpi.h>

#define XSCALE         20

int main(int argc, char* argv[]) { 
  int         nx, *a, *b, *dist, *displ;
  MPI_Comm    inter_comm;
  int         my_rank, np, rmt_np = atoi(argv[1]);
  char        *prog = argv[2], *prog_argv[2] = {strdupa("0"), NULL};
  double      stime, etime;       

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_spawn(prog, prog_argv, rmt_np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &inter_comm, MPI_ERRCODES_IGNORE);
  if (my_rank) {
    MPI_Scatter(NULL, 1, MPI_INT, NULL, 0, MPI_INT, MPI_PROC_NULL, inter_comm);
    MPI_Scatterv(NULL, NULL, NULL, MPI_INT, NULL, 0, MPI_INT, MPI_PROC_NULL, inter_comm);
    MPI_Gatherv(NULL, 0, MPI_INT, NULL, NULL, NULL, MPI_INT, MPI_PROC_NULL, inter_comm);        
    MPI_Finalize(); 
    return EXIT_SUCCESS; 
  }

  nx = 1<<XSCALE;
  a = (int*)malloc( 2*(nx+rmt_np)*sizeof(int) );
  b = a + nx; 
  dist = b + nx;
  displ = dist + rmt_np;

  for(int i=0; i<nx; i++) a[i] = rand();
  for(int i=0; i<rmt_np; i++) {
    dist[i] = nx/rmt_np;
    if (i < nx%rmt_np) dist[i]++;
  }
  displ[0] = 0;
  for(int i=1; i<rmt_np; i++) displ[i] = displ[i-1] + dist[i-1];

  MPI_Scatter(dist, 1, MPI_INT, NULL, 0, MPI_INT, MPI_ROOT, inter_comm); 
  MPI_Scatterv(a, dist, displ, MPI_INT, NULL, 0, MPI_INT, MPI_ROOT, inter_comm);        
  stime = MPI_Wtime();
  MPI_Gatherv(NULL, 0, MPI_INT, b, dist, displ, MPI_INT, MPI_ROOT, inter_comm);    
  etime = MPI_Wtime();

  int k = 0;
  for(int i=0; i<rmt_np; i++) {
    for(int j=i; j<nx; j+=rmt_np) {
      if (a[j] != b[k]) printf("error: k=%d %d %d\n", k, a[j], b[k]);
      k++;
    }
  } 
  printf("time cost: %10.8f\n", etime - stime);
  free(a);
  MPI_Finalize(); 
  return EXIT_SUCCESS; 
} 
