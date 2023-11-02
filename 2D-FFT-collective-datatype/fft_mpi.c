#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <time.h>
#include <unistd.h>
#include <string.h>

#include <mpi.h>

#define NANO           1000000000
#define Max_Thread_Num 256

#define XSCALE         16
#define YSCALE         8
#define FD             1
#define RD            -1

#define Real     float 
#define MPI_Real MPI_FLOAT

void init2D_mpi(Real *ar, Real *ai, int nx, int disty[], MPI_Comm *comm);
void mFFT1D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int n, int opt);
void FFT2D_mpi(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale, int opt, MPI_Comm *comm, int disty[]);
void transpose(Real *a, Real *b, int lnx, int lny, MPI_Comm *comm);

Real FFT2D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale, int opt);
void init2D(Real *ar, Real *ai, int nx, int ny);
int main(int argc, char* argv[]) { 
    int    i, nx, ny, xscale, yscale;
    Real   *ar, *ai, *br, *bi;
    double stime, etime, ptime;       
    
    MPI_Comm    comm;
    int         my_rank, np, *disty, *distx;
  
    MPI_Init(&argc,&argv);
    stime = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    if ( my_rank==0 ) {
        xscale = XSCALE;
        yscale = YSCALE;
        MPI_Bcast(&xscale, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&yscale, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(&xscale, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&yscale, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    nx = 1<<xscale;
    ny = 1<<yscale;

    Real   *cr, *ci, *dr, *di;
    Real eps = 0, max_eps;
    
    distx = (int*)malloc( 2*np*sizeof(int) );
    for(i=0; i<np; i++) {
        distx[i] = nx/np;
        if(i<nx%np) distx[i]++;
    }
    disty = distx + np;
    for(i=0; i<np; i++) {
        disty[i] = ny/np;
        if(i<ny%np) disty[i]++;
    }
    
    ar = (Real*)malloc( 4*disty[my_rank]*nx*sizeof(Real) );
    ai = ar + disty[my_rank]*nx;
    br = ai + disty[my_rank]*nx;
    bi = br + disty[my_rank]*nx;
 
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    init2D_mpi(ar, ai, nx, disty, &comm);
    MPI_Barrier(comm);
    MPI_Comm_free(&comm);
    stime = MPI_Wtime();
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    FFT2D_mpi(ar, ai, br, bi, xscale, yscale, FD, &comm, disty);
    MPI_Comm_free(&comm);
    etime = MPI_Wtime();
    if (my_rank==0) printf("MPI_2DFFT cost: %f\n", etime - stime);  
    
    cr = (Real*)malloc( 4*nx*ny*sizeof(Real) );
    ci = cr + nx*ny;
    dr = ci + nx*ny;
    di = dr + nx*ny;
    init2D(cr, ci, nx, ny); 
    MPI_Barrier(MPI_COMM_WORLD);
    stime = MPI_Wtime();
    if (my_rank==0) FFT2D(cr, ci, dr, di, xscale, yscale, FD);    
    etime = MPI_Wtime();
    MPI_Bcast(dr, nx*ny, MPI_Real, 0, MPI_COMM_WORLD);
    MPI_Bcast(di, nx*ny, MPI_Real, 0, MPI_COMM_WORLD);

    //test result
    int  j;
    int  n = 0;
    for(i=0; i<my_rank; i++) n += disty[i]; 
    for(i=0; i<disty[my_rank]; i++) {
       for ( j=0; j<nx; j++) {
           if ( fabs(dr[(i+n)*nx+j] - br[i*nx+j])>eps ) eps = fabs(dr[(i+n)*nx+j] - br[i*nx+j]); 
           if ( fabs(di[(i+n)*nx+j] - bi[i*nx+j])>eps ) eps = fabs(di[(i+n)*nx+j] - bi[i*nx+j]); 
       }
    }
    MPI_Reduce(&eps, &max_eps, 1, MPI_Real, MPI_MAX, 0, MPI_COMM_WORLD);    
    if (my_rank==0) printf("2D_FFT eps: %e      cost: %f\n", max_eps, etime - stime);   
    
    free(cr);
    
    free(ar);
    free(distx);
    
    MPI_Finalize(); 
    return EXIT_SUCCESS; 
} 

void FFT2D_mpi(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale, int opt, MPI_Comm *comm, int disty[]) {
    int    np, my_rank, *distx;
    int    i, j, nx, ny;
    Real   *cr, *ci, *dr, *di, *bri, *bii; 
    
    MPI_Comm_size(*comm,&np);
    MPI_Comm_rank(*comm,&my_rank);
    
    nx = 1<<xscale;
    ny = 1<<yscale;
    distx = (int*)malloc( np*sizeof(int) );
    for(i=0; i<np; i++) {
        distx[i] = nx/np;
        if(i<nx%np) distx[i]++;
    }
    cr = (Real*)malloc( 4*distx[my_rank]*ny*sizeof(Real) );
    ci = cr + distx[my_rank]*ny;
    dr = ci + distx[my_rank]*ny;
    di = dr + distx[my_rank]*ny;
    
    mFFT1D(ar, ai, br, bi, xscale, disty[my_rank], FD);    
    
    transpose(br, cr, disty[my_rank], distx[my_rank], comm); 
    transpose(bi, ci, disty[my_rank], distx[my_rank], comm);       
    
    mFFT1D(cr, ci, dr, di, yscale, distx[my_rank], FD);
    
    transpose(dr, br, distx[my_rank], disty[my_rank], comm);
    transpose(di, bi, distx[my_rank], disty[my_rank], comm);       
    
    if ( opt == FD )  for(i=0; i<disty[my_rank]; i++) {
        bri = br + i*nx;
        bii = bi + i*nx;
        for(j=0; j<nx; j++) {
            bri[j] = bri[j] / (nx*ny); 
            bii[j] = bii[j] / (nx*ny);
        }                  
    }
    free(cr);
    free(distx);
    return;
}

void transpose(Real *a, Real *b, int lny, int lnx, MPI_Comm *comm) {
     //a is a 2d array, its lny rows are stored locally
     //b is a 2d array that is transposed from a, its lnx rows are stored locally
    int  np;
    MPI_Comm_size(*comm,&np);

    Real *buf;
	MPI_Datatype stypes[np], rtypes[np];	
    int32_t  scounts[np], sdispl[np], rcounts[np], rdispl[np], nx, ny, offset;
    MPI_Allreduce(&lnx, &nx, 1, MPI_INT32_T, MPI_SUM, *comm);
    MPI_Allreduce(&lny, &ny, 1, MPI_INT32_T, MPI_SUM, *comm);		
    posix_memalign((void**)&buf, getpagesize(), ny*lnx*sizeof(Real));
    
    offset = 0;
    MPI_Allgather(&lnx, 1, MPI_INT32_T, rcounts, 1, MPI_INT32_T, *comm);
    for(int i=0; i<np; i++) {
    	sdispl[i] = sizeof(Real)*offset;
    	scounts[i] = lny;
    	MPI_Datatype T_Vec;
    	MPI_Type_contiguous(rcounts[i], MPI_Real, &T_Vec);
    	MPI_Type_create_resized(T_Vec, 0-sdispl[i], sizeof(Real)*nx, &stypes[i]);
    	MPI_Type_commit(&stypes[i]);
    	MPI_Type_free(&T_Vec);
    	offset += rcounts[i];
	}
    /*for(int i=0; i<np; i++) {
    	sdispl[i] = sizeof(Real)*offset;
    	scounts[i] = lny;
    	MPI_Datatype T_Vec;
    	MPI_Type_contiguous(rcounts[i], MPI_Real, &T_Vec);
    	MPI_Type_create_resized(T_Vec, 0, sizeof(Real)*nx, &stypes[i]);
    	MPI_Type_commit(&stypes[i]);
    	MPI_Type_free(&T_Vec);
    	offset += rcounts[i];
	}*/ 
	offset = 0;
	MPI_Allgather(&lny, 1, MPI_INT32_T, rcounts, 1, MPI_INT32_T, *comm);
    for(int i=0; i<np; i++) {
    	rdispl[i] = sizeof(Real)*offset;
    	rcounts[i] = rcounts[i]*lnx;
    	rtypes[i] = MPI_Real;    	
    	offset += rcounts[i];
	} 
    MPI_Alltoallw(a, scounts, sdispl, stypes, buf,rcounts, rdispl, rtypes, *comm);
	offset = 0;	nx=0;
	MPI_Allgather(&lny, 1, MPI_INT32_T, rcounts, 1, MPI_INT32_T, *comm);
	for(int k=0; k<np; k++){
		for(int i=0; i<rcounts[k]; i++) {
			for(int j=0; j<lnx; j++) {
				b[j*ny+i+nx] =  buf[offset + i*lnx + j];
			} 
		}
		nx += rcounts[k];			
		offset += rcounts[k]*lnx;		
	}	
	free(buf); 	
}

void init2D_mpi(Real *ar, Real *ai, int nx, int disty[], MPI_Comm *comm) {
     MPI_Status status;
     int my_rank, np, i, j, t, ny;
     Real *ari, *aii;
     
     MPI_Comm_size(*comm,&np);
     MPI_Comm_rank(*comm,&my_rank);
     
     if ( my_rank>0 ) {
        MPI_Recv(ar, disty[my_rank]*nx, MPI_Real, 0, 10, *comm, &status);
        MPI_Recv(ai, disty[my_rank]*nx, MPI_Real, 0, 20, *comm, &status);
        return;
     }

     ny = disty[0];
     for(t=1; t<np; t++) {
          for(i=0; i<disty[t]; i++) {
              ari = ar + i*nx;
              aii = ai + i*nx;
              for(j=0; j<nx; j++){
                 ari[j] = (j + 1) * (1 + ny);
                 aii[j] = (nx - j) * (1 + ny);
              }
              ny++;              
          }
          MPI_Send(ar, disty[t]*nx, MPI_Real, t, 10, *comm);
          MPI_Send(ai, disty[t]*nx, MPI_Real, t, 20, *comm);
     }
     
     for(i=0; i<disty[my_rank]; i++) {
              ari = ar + i*nx;
              aii = ai + i*nx;
              for(j=0; j<nx; j++){
                 ari[j] = (j + 1) * (1 + i);
                 aii[j] = (nx - j) * (1 + i);
              }
     }
     return;
}

void   mFFT1D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int n, int opt) { 
    //对2D数组的每一行分别进行快速傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部
    //br, bi分别是b的实部和虚部
    //2^xscale是2D数组的列数；n是2D数组的行数
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT    
    double thet, *wr, *wi, wbr, wbi;
    int    nx, i, j, t, k1, k2, kw;
    Real   *ari, *aii, *bri, *bii; 
    int    *brp, *k, *kt, *kj;
    
    nx = 1<<xscale;  
    brp = (int*)malloc( (nx+3*xscale*(nx>>1))*sizeof(int));
    wr = (double*)malloc( nx*sizeof(double));
    wi = wr + (nx>>1); 
    
    for (i = 0; i < nx; i++) { 
        brp[i] = i&1;
        for (t = 1; t < xscale; t++) brp[i] = (brp[i] << 1) + ((i>>t) & 1);
    }
    k = brp + nx;
    for(t=1; t<=xscale; t++) {
        for(j=0; j<nx/2; j++) {
            k[1] = ((j>>(t-1))<<t) + (j&((1<<(t-1)) - 1));
            k[2] = k[1] + (1<<(t-1));
            k[0] = (j&((1<<(t-1)) - 1))*(1<<(xscale-t));
            k += 3;
        }     
    }
    
    if ( opt == FD ) 
         thet = -4*asin(1.0)/nx; 
    else 
         thet = 4*asin(1.0)/nx;
    for (i=0; i<nx/2; i++) { 
         wr[i] = cos(thet*i); 
         wi[i] = sin(thet*i); 
    } 
    
    for(i=0; i<n; i++) {
       ari = ar + i*nx;
       aii = ai + i*nx;
       bri = br + i*nx;
       bii = bi + i*nx;
       for(j=0; j<nx; j++) {
          bri[j] = ari[brp[j]];
          bii[j] = aii[brp[j]];
       } 
    }

    kt = brp + nx;    
    for(t=1; t<=xscale; t++) {       
       for(i=0; i<n; i++) {
         kj = kt;
         bri = br + i*nx;
         bii = bi + i*nx;
         for(j=0; j<nx/2; j++) {
            k1 = kj[1];
            k2 = kj[2];
            kw = kj[0];
            wbr = wr[kw] * bri[k2] - wi[kw] * bii[k2];
            wbi = wi[kw] * bri[k2] + wr[kw] * bii[k2];
            bri[k2] = bri[k1] - wbr; 
            bii[k2] = bii[k1] - wbi;                  
            bri[k1] = bri[k1] + wbr; 
            bii[k1] = bii[k1] + wbi;
            kj += 3;
         }     
       }
       kt += 3*(nx>>1);
    } 
      
    free(brp);
    free(wr);
    
    return;
} 

Real FFT2D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale, int opt) { 
    // 快速2D傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部
    //br, bi分别是b的实部和虚部
    //2^xscale是2D数组的列数；2^yscale是2D数组的行数
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT    
    int    i, j, nx, ny;
    Real   *cr, *ci, *bri, *bii; 
    
    struct timespec ts,te;
    
    clock_gettime(CLOCK_REALTIME, &ts); 
    nx = 1<<xscale;
    ny = 1<<yscale;
    cr = (Real*)malloc( 2*nx*ny*sizeof(Real) );
    ci = cr + nx*ny;

    mFFT1D(ar, ai, cr, ci, xscale, ny, opt);    
    for(i=0; i<ny; i++) 
        for(j=0; j<nx; j++) {
           br[j*ny+i] = cr[i*nx+j];
           bi[j*ny+i] = ci[i*nx+j];
        }
    
    mFFT1D(br, bi, cr, ci, yscale, nx, opt);
    for(i=0; i<nx; i++) 
        for(j=0; j<ny; j++) {
           br[j*nx+i] = cr[i*ny+j];
           bi[j*nx+i] = ci[i*ny+j];
        }
     
    if ( opt == FD ) for(i=0; i<ny; i++) {
        bri = br + i*nx;
        bii = bi + i*nx;
        for(j=0; j<nx; j++) {
            bri[j] = bri[j] / (nx*ny); 
            bii[j] = bii[j] / (nx*ny);
        }                  
    } 
      
    free(cr);
    clock_gettime(CLOCK_REALTIME, &te);
    return (te.tv_sec - ts.tv_sec + (double)(te.tv_nsec-ts.tv_nsec)/NANO);
}

void init2D(Real *ar, Real *ai, int nx, int ny){
     int i, j;
     Real *ari, *aii;
          
     for(i=0; i<ny; i++) {
        ari = ar + i*nx;
        aii = ai + i*nx;
        for(j=0; j<nx; j++){
            ari[j] = (j + 1) * (1 + i);
            aii[j] = (nx - j) * (1 + i);
        }
     }     
     return;
}

