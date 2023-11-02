#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <math.h>

#include "fio.h"
#define Real           double
#define MPI_Real       MPI_DOUBLE
#define TAGBD          2
#define TAGFC          3

void execute_task(Real *body, Real *force, int size) {
     Real fac, fx, fy, fz;
     Real dx, dy, dz, sq, dist; 
     int  i, j, bi,bj,fi,fj;
     
     for ( i=0; i<3*size; i++ ) force[i] = 0;
     for ( i=0; i<size; i++ ) {
         bi = (i<<2);              
         fi = bi-i;              
         for ( j=i+1; j<size; j++ )  {
             bj = (j<<2);
             fj = bj-j;
             //Calculate force between particle i and j according to Newton's Law
             dx = body[bi+1] - body[bj+1];
             dy = body[bi+2] - body[bj+2];
             dz = body[bi+3] - body[bj+3];
             sq = dx*dx + dy*dy + dz*dz;
             dist = sqrt(sq);
             fac = G * body[bi] * body[bj] / ( dist * sq );
             fx = fac * dx;
             fy = fac * dy;
             fz = fac * dz;
             //Add in force and opposite force to particle i and j
             force[fi] -= fx;
             force[fi+1] -= fy;
             force[fi+2] -= fz;
             force[fj] += fx;
             force[fj+1] += fy;
             force[fj+2] += fz;
         }
     }
     return; 
}

void execute_semi_task(Real *lbody, Real *force, int lsize, Real *rbody, int rsize) {
     Real fac, fx, fy, fz;
     Real dx, dy, dz, sq, dist; 
     int  i, j, bi,bj,fi;
     
     for ( i=0; i<lsize; i++ ) {
         bi = (i<<2);            
         fi = bi-i;              
         for ( j=0; j<rsize; j++ )  {
             bj = (j<<2);//4*j;
             //Calculate force between particle i and j according to Newton's Law
             dx = lbody[bi+1] - rbody[bj+1];
             dy = lbody[bi+2] - rbody[bj+2];
             dz = lbody[bi+3] - rbody[bj+3];
             sq = dx*dx + dy*dy + dz*dz;
             dist = sqrt(sq);
             fac = G * lbody[bi] * rbody[bj] / ( dist * sq );
             fx = fac * dx;
             fy = fac * dy;
             fz = fac * dz;
             //Add in force and opposite force to particle i 
             force[fi] -= fx;
             force[fi+1] -= fy;
             force[fi+2] -= fz;
         }
     }
     return; 
}

void  execute_task(Real *lbody, Real *force, int lsize, Real *rbody, Real *rforce, int rsize) {
     Real fac, fx, fy, fz;
     Real dx, dy, dz, sq, dist; 
     int  i, j, bi,bj,fi,fj;     
     
     for ( j=0; j<rsize*3; j++ ) rforce[j] = 0;
     for ( i=0; i<lsize; i++ ) {
         bi = (i<<2);              
         fi = bi-i;              
         for ( j=0; j<rsize; j++ )  {
             bj = (j<<2);
             fj = bj-j;
             //Calculate force between particle i and j according to Newton's Law
             dx = lbody[bi+1] - rbody[bj+1];
             dy = lbody[bi+2] - rbody[bj+2];
             dz = lbody[bi+3] - rbody[bj+3];
             sq = dx*dx + dy*dy + dz*dz;
             dist = sqrt(sq);
             fac = G * lbody[bi] * rbody[bj] / ( dist * sq );
             fx = fac * dx;
             fy = fac * dy;
             fz = fac * dz;
             //Add in force and opposite force to particle i 
             force[fi] -= fx;
             force[fi+1] -= fy;
             force[fi+2] -= fz;
             rforce[fj] += fx;
             rforce[fj+1] += fy;
             rforce[fj+2] += fz;
         }
     }
     return;
}
void swap(Real** rbd_nxt, Real** rbd) {
Real *temp = *rbd_nxt;
    *rbd_nxt = *rbd;
    *rbd = temp;
}

void optimized_nbody(Real *pbd, int time_steps, int64_t *dist) {
    MPI_Status  sta[3]; 
    MPI_Request req[3];
    int  np, my_rank;
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    
    int lnb[np], rnb[np]; //lnb:left neighbor; rnb:right neighbor
    int ncp = (np>>1) + 1;//ncp:number of computation partitionings
    for (int i=0; i<ncp; i++) {
        lnb[i] = my_rank - i;
        rnb[i] = my_rank + i;
        if (lnb[i]<0) lnb[i] += np;
        if (rnb[i]>=np) rnb[i] -= np;
    }      
    ncp = (np>>1) + (np&1);
    
    int  t = 0;
    Real *rbd = (Real*)malloc(8*dist[0]*sizeof(Real));
    Real *pfc = (Real*)malloc(9*dist[0]*sizeof(Real));
    Real *lfc = pfc + 3*dist[0], *rfc = pfc + 6*dist[0];
    Real *rbd_nxt = rbd + 4*dist[0];
    while ( t<time_steps){
        //MPI_Isend(pbd, dist[my_rank]*4, MPI_Real, lnb[1], TAGBD, MPI_COMM_WORLD, &req[0]);
        //MPI_Irecv(rbd_nxt, dist[0]*4, MPI_Real, rnb[1], TAGBD, MPI_COMM_WORLD, &req[1]);
        execute_task(pbd, pfc, dist[my_rank]); 
        //MPI_Waitall(2, req, sta);
        MPI_Sendrecv(pbd, dist[my_rank]*4, MPI_Real, lnb[1], TAGBD, rbd_nxt, dist[0]*4, MPI_Real, rnb[1], TAGBD, MPI_COMM_WORLD, sta);
        swap(&rbd_nxt,&rbd);
        for (int i=1; i<ncp; i++) {           
            if (i<ncp-1 || ((np&1)==0)) {
                //MPI_Isend(pbd, dist[my_rank]*4, MPI_Real, lnb[i+1], TAGBD, MPI_COMM_WORLD, &req[0]);
                //MPI_Irecv(rbd_nxt, dist[0]*4, MPI_Real, rnb[i+1], TAGBD, MPI_COMM_WORLD, &req[1]);
                MPI_Sendrecv(pbd, dist[my_rank]*4, MPI_Real, lnb[i+1], TAGBD, rbd_nxt, dist[0]*4, MPI_Real, rnb[i+1], TAGBD, MPI_COMM_WORLD, sta);
            }
            
            execute_task(pbd, pfc, dist[my_rank], rbd, rfc, dist[rnb[i]]);
            //MPI_Isend(rfc, dist[rnb[i]]*3, MPI_Real, rnb[i], TAGFC, MPI_COMM_WORLD, &req[2]);
            //MPI_Recv(lfc, dist[my_rank]*3, MPI_Real, lnb[i], TAGFC, MPI_COMM_WORLD, &sta[0]); 
            MPI_Sendrecv(rfc, dist[rnb[i]]*3, MPI_Real, rnb[i], TAGFC, lfc, dist[my_rank]*3, MPI_Real, lnb[i], TAGFC, MPI_COMM_WORLD, sta); 
            for(int j=0; j<dist[my_rank]*3; j++ ) pfc[j] += lfc[j];

            if (i<ncp-1 || ((np&1)==0)) {
                //MPI_Waitall(3, req, sta);
                swap(&rbd_nxt,&rbd);
            } else {
                //MPI_Wait(&req[2], &sta[2]);
            }
        }
        if ((np&1)==0) execute_semi_task(pbd, pfc, dist[my_rank], rbd, dist[rnb[ncp]]);  
        
        for (int i=0; i<dist[my_rank]; i++ ){ 
            int bi = (i<<2); 
            int fi = bi-i;
            pbd[bi+1] += pfc[fi] / pbd[bi];
            pbd[bi+2] += pfc[fi+1] / pbd[bi];
            pbd[bi+3] += pfc[fi+2] / pbd[bi];
        }         
        t++;
    }          
      
    free(rbd);
    free(pfc);
    return;
}
int main (int argc, char* argv[]) {
    int32_t     my_rank, np;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); 
    
    FILE_SEG fseg;
    Real *pbd;
    
    int32_t steps = (1<<atoi(argv[2])); 
    int64_t size = ((int64_t)1<<atoi(argv[1])), l_size=size/np, sizes[np], offset;
    if(my_rank<size%np) l_size ++;
    MPI_Scan(&l_size, &offset, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allgather(&l_size, 1, MPI_INT64_T, sizes, 1, MPI_INT64_T, MPI_COMM_WORLD); 
    fseg.offset = 4*(offset - l_size)*sizeof(Real);
    fseg.width = 4*l_size*sizeof(Real);
    fseg.stride = 4*size*sizeof(Real);        
    
    if (posix_memalign((void**)&pbd, getpagesize(), 4*l_size*sizeof(Real))) {
        perror("posix_memalign"); 
        return EXIT_SUCCESS;
    }
    input_data(pbd, 4*l_size*sizeof(Real), fseg);
    
    optimized_nbody(pbd, steps, sizes);
    output_data(pbd, 4*l_size*sizeof(Real), fseg);
    
    MPI_Finalize();
    free(pbd);
     
    return EXIT_SUCCESS;
}
