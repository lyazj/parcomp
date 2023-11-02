#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ser_fio.h"
#define FD (1<<10)
#define RD (1<<11)
#define Real float 

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

double FFT2D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int yscale, int opt) { 
    // 快速2D傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部
    //br, bi分别是b的实部和虚部
    //2^xscale是2D数组的列数；2^yscale是2D数组的行数
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT
    
    int    i, j, nx, ny;
    Real   *cr, *ci, *bri, *bii; 
    
    //struct timespec ts,te;
    
    //clock_gettime(CLOCK_REALTIME, &ts); 
    nx = 1<<xscale;
    ny = 1<<yscale;
    cr = (Real*)malloc( 2*nx*ny*sizeof(Real) );
    ci = cr + nx*ny;

    //mFFT1D(ar, ai, cr, ci, xscale, ny, opt);  
	mFFT1D(ar, ai, cr, ci, yscale, nx, opt);  
 	for(i=0; i<nx; i++) 
        for(j=0; j<ny; j++) {
           br[j*nx+i] = cr[i*ny+j];
           bi[j*nx+i] = ci[i*ny+j];
        }
    
    //mFFT1D(br, bi, cr, ci, yscale, nx, opt);
    mFFT1D(br, bi, cr, ci, xscale, ny, opt);
   for(i=0; i<ny; i++) 
        for(j=0; j<nx; j++) {
           br[j*ny+i] = cr[i*nx+j];
           bi[j*ny+i] = ci[i*nx+j];
        }
     
    if ( opt == RD ) for(i=0; i<ny; i++) {
        bri = br + i*nx;
        bii = bi + i*nx;
        for(j=0; j<nx; j++) {
            bri[j] = bri[j] / (nx*ny); 
            bii[j] = bii[j] / (nx*ny);
        }                  
    } 
      
    free(cr);
    return 0;
}

Real   *ar, *ai, *br, *bi;

int main (int argc, char* argv[]) {
	int64_t i, size = ((int64_t)1<<atoll(argv[1]))*(1<<atoi(argv[2]));
	
    if (posix_memalign((void**)&ar, getpagesize(), size*sizeof(Real))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    input_data(ar, sizeof(Real)*size);
    
    if (posix_memalign((void**)&ai, getpagesize(), size*sizeof(Real))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    if (posix_memalign((void**)&br, getpagesize(), size*sizeof(Real))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    if (posix_memalign((void**)&bi, getpagesize(), size*sizeof(Real))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
    memset(ai, 0, size*sizeof(Real));
    FFT2D(ar, ai, br, bi, atoi(argv[1]), atoi(argv[2]), FD);
        
    free(ar);
    free(ai);
    float *pc;
    if (posix_memalign((void**)&pc, getpagesize(), (size<<1)*sizeof(Real))) {
		perror("posix_memalign: serial"); 
		return EXIT_SUCCESS;
	}
	for(int64_t i=0; i<size; i++) {
		pc[i<<1]=br[i];
		pc[(i<<1)+1]=bi[i];
	}
	output_data(pc, (size<<1)*sizeof(Real)); 
	free(pc);
    free(br);
    free(bi);

    return EXIT_SUCCESS;
}
