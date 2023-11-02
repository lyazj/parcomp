#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include "fio.h"
#define Real float 

void   mFFT1D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int n) { 
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
    
    thet = -4*asin(1.0)/nx;
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

double FFT2D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale) { 
    // 快速2D傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部
    //br, bi分别是b的实部和虚部
    //2^xscale是2D数组的列数；2^yscale是2D数组的行数
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT
    
    int    i, j, nx, ny;
    Real   *cr, *ci, *bri, *bii; 
    
    //struct timespec ts,te;
    
    nx = 1<<xscale;
    ny = 1<<yscale;
    cr = (Real*)malloc( 2*nx*ny*sizeof(Real) );
    ci = cr + nx*ny;

    mFFT1D(ar, ai, cr, ci, yscale, nx);    
    for(i=0; i<nx; i++) 
        for(j=0; j<ny; j++) {
           br[j*nx+i] = cr[i*ny+j];
           bi[j*nx+i] = ci[i*ny+j];
        }
    
    mFFT1D(br, bi, cr, ci, xscale, ny);
    for(i=0; i<ny; i++) 
        for(j=0; j<nx; j++) {
           br[j*ny+i] = cr[i*nx+j];
           bi[j*ny+i] = ci[i*nx+j];
        }
     
      
    free(cr);
    return 0;
}

int main (int argc, char* argv[]) {
    MPI_Comm    comm;
    MPI_Status  status;
	int32_t     my_rank, np;
	MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&np);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank); 

	FILE_SEG fseg;
	float   *ar, *ai, *br, *bi;
	
	int64_t  size = atoll(argv[1]), l_size, sizes[np], offset;
    size = ((int64_t)1<<size);
    l_size = size / np;
    if (my_rank < size % np) l_size ++;
    size *= (1<<atoi(argv[2]));
    l_size *= (1<<atoi(argv[2]));
    MPI_Scan(&l_size, &offset, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allgather(&l_size, 1, MPI_INT64_T, sizes, 1, MPI_INT64_T, MPI_COMM_WORLD); 
	fseg.offset = (offset - l_size)*sizeof(float);
	fseg.width = l_size*sizeof(float);
	fseg.stride = size*sizeof(float);		
        
    if (posix_memalign((void**)&ar, getpagesize(), l_size*sizeof(float))) {
		perror("posix_memalign: data"); 
		return EXIT_SUCCESS;
	}
	input_data(ar, l_size*sizeof(float), fseg);

    int32_t recvcounts[np], displs[np];
	for(int i=0; i<np; i++) recvcounts[i] = sizes[i];
    displs[0] = 0;
    for(int i=1; i<np; i++) displs[i] = displs[i-1] + recvcounts[i-1];
    if (posix_memalign((void**)&br, getpagesize(), size*sizeof(float))) {
		perror("posix_memalign"); 
		return EXIT_SUCCESS;
	}
    MPI_Allgatherv(ar, recvcounts[my_rank], MPI_FLOAT, br, recvcounts, displs, MPI_FLOAT, MPI_COMM_WORLD);
    free(ar);
    ar = br;

    if (posix_memalign((void**)&ai, getpagesize(), size*sizeof(float))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    if (posix_memalign((void**)&br, getpagesize(), size*sizeof(float))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    if (posix_memalign((void**)&bi, getpagesize(), size*sizeof(float))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    memset(ai, 0, size*sizeof(float));
    FFT2D(ar, ai, br, bi, atoi(argv[1]), atoi(argv[2]));
    
    float *pc;
    if (posix_memalign((void**)&pc, getpagesize(), (l_size<<1)*sizeof(float))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
	offset -= l_size;
	for(int64_t i=0; i<l_size; i++) {
		pc[i<<1]=br[i+offset];
		pc[(i<<1)+1]=bi[i+offset];
	}

    l_size = (l_size<<1);
	MPI_Scan(&l_size, &offset, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
	fseg.offset = (offset - l_size)*sizeof(float);
	fseg.width = l_size*sizeof(float);
	fseg.stride = (size<<1)*sizeof(float);
	output_data(pc, sizeof(float)*l_size, fseg);
	
	free(ar);
	free(ai);
	free(br);
	free(bi);
	free(pc); 

    MPI_Finalize();
    return EXIT_SUCCESS;
}

