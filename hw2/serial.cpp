#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h> 

// #include "ser_fio.h"
#include "fio.h"
#define REAL double

int64_t size=0;
int32_t T=0;
REAL *pbd;
REAL *pfc;

                                                                   
int main(int argc, char** argv ) {     
    int64_t i, j;
    int32_t t = 0;
    
    size = atoll(argv[1]);
    T = (1<<atoi(argv[2]));
    size = ((int64_t)1<<size);
    REAL *pbd = (REAL*)malloc(4*size*sizeof(REAL));
    input_data(pbd, 4*size*sizeof(REAL));
    
    pfc = (REAL*)malloc(3*size*sizeof(REAL));
    memset(pfc, 0, 3*size*sizeof(REAL));
           
    while ( t<T){
        /*  Loop over points calculating force between each pair.*/
        for ( i=0; i<size; i++ ) {
            int64_t bi = (i<<2);              
            int64_t fi = bi - i;              
            for ( j=i+1; j<size; j++ )  {
                int64_t bj = (j<<2);
                int64_t fj = bj - j;
                /*Calculate force between particle i and j according to Newton's Law*/
                REAL dx = pbd[bi+1] - pbd[bj+1];
                REAL dy = pbd[bi+2] - pbd[bj+2];
                REAL dz = pbd[bi+3] - pbd[bj+3];
                REAL sq = dx*dx + dy*dy + dz*dz;
                REAL dist = sqrt(sq);
                REAL fac = G * pbd[bi] * pbd[bj] / ( dist * sq );
                REAL fx = fac * dx;
                REAL fy = fac * dy;
                REAL fz = fac * dz;
                /*Add in force and opposite force to particle i and j */
                pfc[fi] -= fx;
                pfc[fi+1] -= fy;
                pfc[fi+2] -= fz;
                pfc[fj] += fx;
                pfc[fj+1] += fy;
                pfc[fj+2] += fz;
            }
        }
        for ( i=0; i<size; i++ ) { 
            int64_t bi = (i<<2);              
            int64_t fi = bi - i;             
            pbd[bi+1] = pbd[bi+1] + pfc[fi] / pbd[bi]; 
			pfc[fi] = 0;
            pbd[bi+2] = pbd[bi+2] + pfc[fi+1] / pbd[bi];
            pfc[fi+1] = 0;
            pbd[bi+3] = pbd[bi+3] + pfc[fi+2] / pbd[bi];
            pfc[fi+2] = 0;
        }
        t++;
    }
    
    output_data(pbd, 4*size*sizeof(REAL));
    free(pbd);
    free(pfc);
    
    return EXIT_SUCCESS;
}

