#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#include <mpi.h>

#define NANO           1000000000
#define Max_Thread_Num 256

#define XSCALE         16
#define YSCALE         8
#define FD             1
#define RD            -1

#define Real     float
#define MPI_Real MPI_FLOAT

void init1D_mpi(Real *ar, Real *ai, int nx, MPI_Comm *comm, int distx[]);
void FFT1D_mpi(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int opt, MPI_Comm *comm, int distx[]);

void init1D(Real *ar, Real *ai, int nx);
Real FFT1D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int opt);
int main(int argc, char* argv[]) {

    int    i, nx, ny, xscale, yscale;
    Real   *ar, *ai, *br, *bi;
    double stime, etime, ptime;

    MPI_Comm    comm;
    int         my_rank, np, *disty, *distx;

    MPI_Init(&argc,&argv);

    // ******************** 学生增补代码始 ********************

    MPI_Comm the_big_world = MPI_COMM_NULL;
    if(argc != 1) {
        // 校验命令行参数个数
        if(argc != 2) {
            fprintf(stderr, "ERROR: expect 1 or 2 arguments, got %d\n", argc);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 特殊标记：增补出的进程 np_exp 参数置零
        int np_exp = atoi(argv[1]);
        if(np_exp == 0) {
            MPI_Comm parent; MPI_Comm_get_parent(&parent);
            MPI_Intercomm_merge(parent, 1, &the_big_world);
            goto NO_NEED_TO_SPAWN;
        }

        // np_exp 少于当前已有进程数，说明出现了逻辑错误
        MPI_Comm_size(MPI_COMM_WORLD, &np);
        if(np_exp < np) {
            fprintf(stderr, "ERROR: too many processes created\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 不需增补时，跳过该步骤
        if(np_exp == np) goto NO_NEED_TO_SPAWN;

        // 现在已有 np 个进程，需要增补 (np_exp - np) 个进程
        char *spawn_argv[] = { strdupa("0"), NULL };
        int spawn_code[np_exp - np];
        MPI_Comm child;
        MPI_Comm_spawn(argv[0], spawn_argv, np_exp - np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &child, spawn_code);
        MPI_Intercomm_merge(child, 0, &the_big_world);
    }
NO_NEED_TO_SPAWN:
    // 没有发生进程增补时全局通信子 fallback 至 MPI_COMM_WORLD
    if(the_big_world == MPI_COMM_NULL) MPI_Comm_dup(MPI_COMM_WORLD, &the_big_world);

    // 已将下文所有 MPI_COMM_WORLD 替换为 the_big_world
    // ******************** 学生增补代码终 ********************

    stime = MPI_Wtime();
    MPI_Comm_size(the_big_world,&np);
    MPI_Comm_rank(the_big_world,&my_rank);
    if ( my_rank==0 ) {
        xscale = XSCALE;
        yscale = YSCALE;
        MPI_Bcast(&xscale, 1, MPI_INT, 0, the_big_world);
        MPI_Bcast(&yscale, 1, MPI_INT, 0, the_big_world);
    }
    else {
        MPI_Bcast(&xscale, 1, MPI_INT, 0, the_big_world);
        MPI_Bcast(&yscale, 1, MPI_INT, 0, the_big_world);
    }
    nx = 1<<xscale;
    //perform 1D FFT
    distx = (int*)malloc( np*sizeof(int) );
    for(i=0; i<np; i++) {
             distx[i] = nx/np;
             if(i<nx%np) distx[i]++;
    }
    ar = (Real*)malloc( 4*distx[my_rank]*sizeof(Real) );
    ai = ar + distx[my_rank];
    br = ai + distx[my_rank];
    bi = br + distx[my_rank];
    MPI_Comm_dup(the_big_world, &comm);
    init1D_mpi(ar, ai, nx, &comm, distx);
    MPI_Comm_free(&comm);
    MPI_Barrier(the_big_world);
    stime = MPI_Wtime();
    MPI_Comm_dup(the_big_world, &comm);
    FFT1D_mpi(ar, ai, br, bi, xscale, FD, &comm, distx);
    MPI_Comm_free(&comm);
    etime = MPI_Wtime();
    if (my_rank==0) printf("MPI_1DFFT cost: %f\n", etime - stime);

    Real   *cr, *ci, *dr, *di;
    cr = (Real*)malloc( 4*nx*sizeof(Real) );
    ci = cr + nx;
    dr = ci + nx;
    di = dr + nx;
    init1D(cr, ci, nx);
    FFT1D(cr, ci, dr, di, xscale, FD);
    //test result
    Real eps = 0, max_eps;
    int  n = 0;
    for(i=0; i<my_rank; i++) n += distx[i];
    for(i=0; i<distx[my_rank]; i++) {
       if ( fabs(dr[i+n] - br[i])>eps ) eps = fabs(dr[i+n] - br[i]);
       if ( fabs(di[i+n] - bi[i])>eps ) eps = fabs(di[i+n] - bi[i]);
     }
    MPI_Reduce(&eps, &max_eps, 1, MPI_Real, MPI_MAX, 0, the_big_world);
    if (my_rank==0) printf("1D_FFT eps: %e\n", max_eps);

    free(cr);
    free(ar);
    free(distx);

    // ******************** 学生增补代码始 ********************

    MPI_Comm_free(&the_big_world);

    // ******************** 学生增补代码终 ********************

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void FFT1D_mpi(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int opt, MPI_Comm *comm, int distx[]) {
    // 快速1D傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部;br, bi分别是b的实部和虚部
    //2^xscale是1D数组的长度
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT
    MPI_Status status;
    MPI_Comm   newcomm;
    int my_rank, np, color, *displ;
    double thet;
    int    nx, i, t, k1, k2, kw, brp, lnx, m, pair, msgLen;
    double *wr, *wi, wbr, wbi;
    Real   *buf, *cr, *ci, temp;

    MPI_Comm_size(*comm,&np);
    MPI_Comm_rank(*comm,&my_rank);
    nx = 1<<xscale;

    if ( my_rank==0) {
         buf = (Real*)malloc( 2*nx*sizeof(Real));
         displ = (int*)malloc( np*sizeof(int));
         displ[0] = 0;
         for(i=1; i<np; i++) displ[i] = displ[i-1] + distx[i-1];
         MPI_Gatherv(ar, distx[my_rank], MPI_Real, buf, distx, displ, MPI_Real, 0, *comm);
         MPI_Gatherv(ai, distx[my_rank], MPI_Real, buf+nx, distx, displ, MPI_Real, 0, *comm);

         for(i=0; i<nx; i++) {
            brp = i&1;
            for (t = 1; t < xscale; t++) brp = (brp << 1) + ((i>>t) & 1);
            if ( brp<=i ) continue;
            temp = buf[brp];
            buf[brp] = buf[i];
            buf[i]   = temp;
            temp = buf[brp+nx];
            buf[brp+nx] = buf[i+nx];
            buf[i+nx]   = temp;
         }

     }
    else {
         MPI_Gatherv(ar, distx[my_rank], MPI_Real, buf, distx, displ, MPI_Real, 0, *comm);
         MPI_Gatherv(ai, distx[my_rank], MPI_Real, buf+nx, distx, displ, MPI_Real, 0, *comm);
    }

    m  = 0;
    while((1<<(m+1))<=np) m++;
    np = 1<<m;
    if ( my_rank<np ) {//perform 1D FFT with 2^t processors
        color = 1;
        MPI_Comm_split(*comm, color, 0, &newcomm);
        lnx = 1<<(xscale - m);
        msgLen = lnx>>1;
        cr = (Real*)malloc( (2*lnx)*sizeof(Real) );
        ci = cr + lnx;
        MPI_Scatter(buf, lnx, MPI_Real, cr, lnx, MPI_Real, 0, newcomm);
        MPI_Scatter(buf+nx, lnx, MPI_Real, ci, lnx, MPI_Real, 0, newcomm);

        wr = (double*)malloc( nx*sizeof(double));
        wi = wr + nx/2;
        if ( opt == FD )
           thet = -4*asin(1.0)/nx;
        else
           thet = 4*asin(1.0)/nx;
        for (i=0; i<nx/2; i++) {
           wr[i] = cos(thet*i);
           wi[i] = sin(thet*i);
        }


        for(t=1; t<=xscale-m; t++) {
           for(i=0; i<lnx/2; i++) {
               k1 = ((i>>(t-1))<<t) + (i&((1<<(t-1)) - 1));
               k2 = k1 + (1<<(t-1));
               kw = (i&((1<<(t-1)) - 1))*(1<<(xscale-t));
               wbr = wr[kw] * cr[k2] - wi[kw] * ci[k2];
               wbi = wi[kw] * cr[k2] + wr[kw] * ci[k2];
               cr[k2] = cr[k1] - wbr;
               ci[k2] = ci[k1] - wbi;
               cr[k1] = cr[k1] + wbr;
               ci[k1] = ci[k1] + wbi;
           }
        }

        for(t=1; t<=m; t++) if ( (my_rank&((1<<t)-1)) < (1<<(t-1)) ){
            pair = my_rank + (1<<(t-1));
            MPI_Sendrecv_replace(cr+msgLen, msgLen, MPI_Real, pair, 10, pair, 20, newcomm, &status);
            MPI_Sendrecv_replace(ci+msgLen, msgLen, MPI_Real, pair, 30, pair, 40, newcomm, &status);

            for(i=0; i<lnx/2; i++) {
               k1 = i;
               k2 = k1 + (lnx>>1);
               kw = ((i+lnx*my_rank)&((1<<(xscale-m+t))-1))<<(m-t);
               //printf("t=%d my_rank=%d pair=%d: k1=%d k2=%d kw=%d\n",t, my_rank, pair, k1, k2, kw);
               wbr = wr[kw] * cr[k2] - wi[kw] * ci[k2];
               wbi = wi[kw] * cr[k2] + wr[kw] * ci[k2];
               cr[k2] = cr[k1] - wbr;
               ci[k2] = ci[k1] - wbi;
               cr[k1] = cr[k1] + wbr;
               ci[k1] = ci[k1] + wbi;
           }

            MPI_Sendrecv_replace(cr+msgLen, msgLen, MPI_Real, pair, 10, pair, 20, newcomm, &status);
            MPI_Sendrecv_replace(ci+msgLen, msgLen, MPI_Real, pair, 30, pair, 40, newcomm, &status);
        }
        else {
            pair = my_rank - (1<<(t-1));
            MPI_Sendrecv_replace(cr, msgLen, MPI_Real, pair, 20, pair, 10, newcomm, &status);
            MPI_Sendrecv_replace(ci, msgLen, MPI_Real, pair, 40, pair, 30, newcomm, &status);

            for(i=0; i<lnx/2; i++) {
               k1 = i;
               k2 = k1 + (lnx>>1);
               kw = (((i+lnx*my_rank)&((1<<(xscale-m+t))-1)) - (1<<(xscale-m+t-1)) + (lnx>>1))<<(m-t);
               //printf("t=%d my_rank=%d pair=%d: k1=%d k2=%d kw=%d\n",t, my_rank, pair, k1, k2, kw);

               wbr = wr[kw] * cr[k2] - wi[kw] * ci[k2];
               wbi = wi[kw] * cr[k2] + wr[kw] * ci[k2];
               cr[k2] = cr[k1] - wbr;
               ci[k2] = ci[k1] - wbi;
               cr[k1] = cr[k1] + wbr;
               ci[k1] = ci[k1] + wbi;
            }

            MPI_Sendrecv_replace(cr, msgLen, MPI_Real, pair, 20, pair, 10, newcomm, &status);
            MPI_Sendrecv_replace(ci, msgLen, MPI_Real, pair, 40, pair, 30, newcomm, &status);
        }
        if ( opt==FD) for(i=0; i<lnx; i++) {
           cr[i] /= nx;
           ci[i] /= nx;
        }


        MPI_Gather(cr, lnx, MPI_Real, buf, lnx, MPI_Real, 0, newcomm);
        MPI_Gather(ci, lnx, MPI_Real, buf+nx, lnx, MPI_Real, 0, newcomm);

        free(cr);
        free(wr);
    }
    else {
         color = 0;
         MPI_Comm_split(*comm, color, 0, &newcomm);
    }
    MPI_Comm_free(&newcomm);
    MPI_Barrier(*comm);


    if ( my_rank==0) {
         MPI_Scatterv(buf, distx, displ, MPI_Real, br,  distx[my_rank], MPI_Real, 0, *comm);
         MPI_Scatterv(buf+nx, distx, displ, MPI_Real, bi,  distx[my_rank], MPI_Real, 0, *comm);
         free(buf);
         free(displ);
     }
    else {
         MPI_Scatterv(buf, distx, displ, MPI_Real, br,  distx[my_rank], MPI_Real, 0, *comm);
         MPI_Scatterv(buf+nx, distx, displ, MPI_Real, bi,  distx[my_rank], MPI_Real, 0, *comm);
    }

    return;
}

void init1D_mpi(Real *ar, Real *ai, int nx, MPI_Comm *comm, int distx[]) {
     MPI_Status status;
     int my_rank, np, i, j, t, ny;

     MPI_Comm_size(*comm,&np);
     MPI_Comm_rank(*comm,&my_rank);

     if ( my_rank>0 ) {
        MPI_Recv(ar, distx[my_rank], MPI_Real, 0, 10, *comm, &status);
        MPI_Recv(ai, distx[my_rank], MPI_Real, 0, 20, *comm, &status);
        return;
     }

     ny = distx[my_rank];
     for(t=1; t<np; t++) {
          for(i=0; i<distx[t]; i++) {
              ar[i] = ny + 1;
              ai[i] = nx - ny;
              ny++;
          }
          MPI_Send(ar, distx[t], MPI_Real, t, 10, *comm);
          MPI_Send(ai, distx[t], MPI_Real, t, 20, *comm);
     }

     for(i=0; i<distx[my_rank]; i++) {
        ar[i] = i + 1;
        ai[i] = nx - i;
     }
     return;
}

Real FFT1D(Real *ar, Real *ai, Real *br, Real *bi, int xscale, int opt) {
    // 快速1D傅立叶变换，将复数a变换后保存在b中
    //ar, ai分别是a的实部和虚部;br, bi分别是b的实部和虚部
    //2^xscale是1D数组的长度
    //opt=FD---forword-direction FFT; opt=RD---reverse-direction FFT
    double thet;
    int    nx, i, t, k1, k2, kw;
    double *wr, *wi, wbr, wbi;
    int    *brp;

    struct timespec ts,te;

    clock_gettime(CLOCK_REALTIME, &ts);
    nx = 1<<xscale;

    brp = (int*)malloc( nx*sizeof(int));
    wr = (double*)malloc( nx*sizeof(double));
    wi = wr + nx/2;

    for (i = 0; i < (1<<xscale); i++) {
        brp[i] = i&1;
        for (t = 1; t < xscale; t++) brp[i] = (brp[i] << 1) + ((i>>t) & 1);
     }

    if ( opt == FD )
         thet = -4*asin(1.0)/nx;
    else
         thet = 4*asin(1.0)/nx;
    for (i=0; i<nx/2; i++) {
         wr[i] = cos(thet*i);
         wi[i] = sin(thet*i);
    }

    for(i=0; i<nx; i++) {
        br[i] = ar[brp[i]];
        bi[i] = ai[brp[i]];
    }

    for(t=1; t<=xscale; t++) {
        for(i=0; i<nx/2; i++) {
            k1 = ((i>>(t-1))<<t) + (i&((1<<(t-1)) - 1));
            k2 = k1 + (1<<(t-1));
            kw = (i&((1<<(t-1)) - 1))*(1<<(xscale-t));
            wbr = wr[kw] * br[k2] - wi[kw] * bi[k2];
            wbi = wi[kw] * br[k2] + wr[kw] * bi[k2];
            br[k2] = br[k1] - wbr;
            bi[k2] = bi[k1] - wbi;
            br[k1] = br[k1] + wbr;
            bi[k1] = bi[k1] + wbi;
        }
    }

    if ( opt == FD ) for(i=0; i<nx; i++) {
            br[i] = br[i] / nx;
            bi[i] = bi[i] / nx;
    }

    free(brp);
    free(wr);
    clock_gettime(CLOCK_REALTIME, &te);
    return (te.tv_sec - ts.tv_sec + (double)(te.tv_nsec-ts.tv_nsec)/NANO);
}

void init1D(Real *ar, Real *ai, int nx){
     int i;

     for(i=0; i<nx; i++) {
        ar[i] = i + 1;
        ai[i] = nx - i;
     }
     return;
}

