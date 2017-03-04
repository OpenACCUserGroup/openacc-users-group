//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB EP code. This C        //
//  version is developed by the Center for Manycore Programming at Seoul   //
//  National University and derived from the serial Fortran versions in    //
//  "NPB3.3-SER" developed by NAS.                                         //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this C version to cmp@aces.snu.ac.kr  //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Jeongho Nah, Gangwon Jo,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
////                                                                         //
////  The OpenACC C version of the NAS EP code is developed by the           //
////  HPCTools Group of University of Houston and derived from the serial    //
////  C version developed by SNU and Fortran versions in "NPB3.3-SER"        //
////  developed by NAS.                                                      //
////                                                                         //
////  Permission to use, copy, distribute and modify this software for any   //
////  purpose with or without fee is hereby granted. This software is        //
////  provided "as is" without express or implied warranty.                  //
////                                                                         //
////  Send comments or suggestions for this OpenACC version to               //
////                      hpctools@cs.uh.edu                                 //
////
////  Information on NPB 3.3, including the technical report, the original   //
////  specifications, source code, results and information on how to submit  //
////  new results, is available at:                                          //
////                                                                         //
////           http://www.nas.nasa.gov/Software/NPB/                         //
////                                                                         //
////-------------------------------------------------------------------------//
//
////-------------------------------------------------------------------------//
//// Authors: Rengan Xu, Sunita Chandrasekaran, Barbara Chapman              //
////-------------------------------------------------------------------------//

//--------------------------------------------------------------------
//      program EMBAR
//--------------------------------------------------------------------
//  This is the OpenACC version of the APP Benchmark 1,
//  the "embarassingly parallel" benchmark.
//
//
//  M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//  numbers.  MK is the Log_2 of the size of each batch of uniform random
//  numbers.  MK can be set for convenience on a given system, since it does
//  not affect the results.
//--------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "type.h"
#include "npbparams.h"
#include "timers.h"
#include "print_results.h"
#include <openacc.h>

#define MAX(X,Y)  (((X) > (Y)) ? (X) : (Y))

int MK;
int MM;
int NN;
double EPSILON;
double A;
double S;
int NK;
int NQ;

int BLKSIZE;

double r23;
double r46;
double t23;
double t46;

inline double randlc_ep( double *x, double a )
{
  //--------------------------------------------------------------------
  //
  //  This routine returns a uniform pseudorandom double precision number in the
  //  range (0, 1) by using the linear congruential generator
  //
  //  x_{k+1} = a x_k  (mod 2^46)
  //
  //  where 0 < x_k < 2^46 and 0 < a < 2^46.  This scheme generates 2^44 numbers
  //  before repeating.  The argument A is the same as 'a' in the above formula,
  //  and X is the same as x_0.  A and X must be odd double precision integers
  //  in the range (1, 2^46).  The returned value randlc_ep is normalized to be
  //  between 0 and 1, i.e. randlc_ep = 2^(-46) * x_1.  X is updated to contain
  //  the new seed x_1, so that subsequent calls to randlc_ep using the same
  //  arguments will generate a continuous sequence.
  //
  //  This routine should produce the same results on any computer with at least
  //  48 mantissa bits in double precision floating point data.  On 64 bit
  //  systems, double precision should be disabled.
  //
  //  David H. Bailey     October 26, 1990
  //
  //--------------------------------------------------------------------

  // r23 = pow(0.5, 23.0);
  ////  pow(0.5, 23.0) = 1.1920928955078125e-07
  // r46 = r23 * r23;
  // t23 = pow(2.0, 23.0);
  ////  pow(2.0, 23.0) = 8.388608e+06
  // t46 = t23 * t23;
  /*
  const double r23 = 1.1920928955078125e-07;
  const double r46 = r23 * r23;
  const double t23 = 8.388608e+06;
  const double t46 = t23 * t23;
   */
  double t1, t2, t3, t4, a1, a2, x1, x2, z;
  double r;

  //--------------------------------------------------------------------
  //  Break A into two parts such that A = 2^23 * A1 + A2.
  //--------------------------------------------------------------------
  t1 = r23 * a;
  a1 = (int) t1;
  a2 = a - t23 * a1;

  //--------------------------------------------------------------------
  //  Break X into two parts such that X = 2^23 * X1 + X2, compute
  //  Z = A1 * X2 + A2 * X1  (mod 2^23), and then
  //  X = 2^23 * Z + A2 * X2  (mod 2^46).
  //--------------------------------------------------------------------
  t1 = r23 * (*x);
  x1 = (int) t1;
  x2 = *x - t23 * x1;
  t1 = a1 * x2 + a2 * x1;
  t2 = (int) (r23 * t1);
  z = t1 - t23 * t2;
  t3 = t23 * z + a2 * x2;
  t4 = (int) (r46 * t3);
  *x = t3 - t46 * t4;
  r = r46 * (*x);

  return r;
}

int main() 
{
  double Mops, t1, t2, t3, t4, x1, x2;
  double sx, sy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    np;
  int    i, ik, kk, l, k, nit;
  int    k_offset, j;
  int verified, timers_enabled;
  double q0, q1, q2, q3, q4, q5, q6, q7, q8, q9;

  MK =  16;
  MM =  (M - MK);
  NN =       (1 << MM);
  EPSILON =  1.0e-8;
  A =        1220703125.0;
  S =        271828183.0;
  NK = 1 << MK;
  NQ = 10;

  //  BLKSIZE = 1792;
  BLKSIZE = 2048;

  r23 = 1.1920928955078125e-07;
  r46 = r23 * r23;
  t23 = 8.388608e+06;
  t46 = t23 * t23;

  double x[2*(1<<16)];
  double q[10]; 
  double *xx, *qq;

  /*variables for inlining vranlc()*/
  double in_t1, in_t2, in_t3, in_t4;
  double in_a1, in_a2, in_x1, in_x2, in_z;

  double tmp_sx, tmp_sy;
  double dum[3] = {1.0, 1.0, 1.0};
  char   size[16];


  int blksize = BLKSIZE;
  int blk, koff, numblks;

  FILE *fp;

  acc_init(acc_device_default);

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = 0;
  } else {
    timers_enabled = 1;
    fclose(fp);
  }

  if (NN < blksize) {
    blksize = NN;
  }
  numblks = ceil( (double)NN / (double) blksize);

  xx = (double*)malloc(blksize*2*NK*sizeof(double));
  qq = (double*)malloc(blksize*NQ*sizeof(double));

  //--------------------------------------------------------------------
  //  Because the size of the problem is too large to store in a 32-bit
  //  integer for some classes, we put it into a string (for printing).
  //  Have to strip off the decimal point put in there by the floating
  //  point print statement (internal file)
  //--------------------------------------------------------------------

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-ACC-C) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = 0;

  //--------------------------------------------------------------------
  //  Compute the number of "batches" of random number pairs generated 
  //  per processor. Adjust if the number of processors does not evenly 
  //  divide the total number
  //--------------------------------------------------------------------

  np = NN; 
  printf("NK=%d NN=%d NQ=%d BLKS=%d NBLKS=%d\n",NK,NN,NQ,blksize,numblks);
  //--------------------------------------------------------------------
  //  Call the random number generator functions and initialize
  //  the x-array to reduce the effects of paging on the timings.
  //  Also, call all mathematical functions that are used. Make
  //  sure these initializations cannot be eliminated as dead code.
  //--------------------------------------------------------------------
#pragma acc data create(xx[0:blksize*2*NK],qq[0:blksize*NQ]) copyout(q[0:NQ])
  {
    vranlc(0, &dum[0], dum[1], &dum[2]);
    dum[0] = randlc_ep(&dum[1], dum[2]);

#ifndef CRPL_COMP
#pragma acc parallel num_gangs((NQ+127)/128) vector_length(128) present(q[0:NQ])
#elif CRPL_COMP == 0
#pragma acc kernels present(q[0:NQ])
#endif
    {
#pragma acc loop gang vector independent
      for (i = 0; i < NQ; i++) {
        q[i] = 0.0;
      }
    }
    Mops = log(sqrt(fabs(MAX(1.0, 1.0))));

    timer_clear(0);
    timer_clear(1);
    timer_clear(2);
    timer_start(0);

    /*this function actullay does nothing, so comment it*/
    //vranlc(0, &t1, A, x);

    //#pragma acc update device(x[0:2*NK])
    //--------------------------------------------------------------------
    //  Compute AN = A ^ (2 * NK) (mod 2^46).
    //--------------------------------------------------------------------

    t1 = A;

    for (i = 0; i < MK + 1; i++) {
      t2 = randlc_ep(&t1, t1);
    }

    an = t1;
    tt = S;
    gc = 0.0;
    sx = 0.0;
    sy = 0.0;
    k_offset = -1;

    for (blk=0; blk < numblks; ++blk) {

      koff = blk*blksize;

      if (koff + blksize > np) {
        blksize = np - (blk*blksize);
      }

#ifndef CRPL_COMP
#pragma acc parallel num_gangs(blksize) vector_length(128) present(qq[0:blksize*NQ])
#elif CRPL_COMP == 0
#pragma acc kernels present(qq[0:blksize*NQ])
#endif
      {
#pragma acc loop gang independent
        for(k=0; k<blksize; k++)
        {
#pragma acc loop vector independent
          for(i=0; i<NQ; i++)
            qq[k*NQ + i] = 0.0;
        }
      }

      /*
#pragma acc parallel num_gangs(blksize) vector_length(128) present(xx[0:blksize*2*NK])
{
 #pragma acc loop gang
 for(k=0; k<blksize; k++)
  {
  #pragma acc loop vector
  for(i=0; i<2*NK; i++)
    xx[k*2*NK + i] = -1.0e99;
  }
 }
       */
      //--------------------------------------------------------------------
      //  Each instance of this loop may be performed independently. We compute
      //  the k offsets separately to take into account the fact that some nodes
      //  have more numbers to generate than others
      //--------------------------------------------------------------------

#ifndef CRPL_COMP
#pragma acc parallel num_gangs((blksize+255)/256) num_workers(1) vector_length(256) \
        present(xx[0:blksize*2*NK],qq[0:blksize*NQ])
#elif CRPL_COMP == 0
#pragma acc kernels present(xx[0:blksize*2*NK],qq[0:blksize*NQ])
#endif
      {
#pragma acc loop gang worker vector reduction(+:sx,sy) independent
        for (k = 1; k <= blksize; k++) {
          kk = k_offset + k + koff;
          t1 = S;
          t2 = an;

          // Find starting seed t1 for this kk.

          for (i = 1; i <= 100; i++) {
            ik = kk / 2;
            if ((2 * ik) != kk)
            {
              in_t1 = r23 * t2;
              in_a1 = (int)in_t1;
              in_a2 = t2 - t23 * in_a1;

              in_t1 = r23 * t1;
              in_x1 = (int)in_t1;
              in_x2 = t1 - t23 * in_x1;
              in_t1 = in_a1 * in_x2 + in_a2 * in_x1;
              in_t2 = (int)(r23 * in_t1);
              in_z = in_t1 - t23 * in_t2;
              in_t3 = t23 * in_z + in_a2 * in_x2;
              in_t4 = (int)(r46 * in_t3);
              t1 = in_t3 - t46 * in_t4;
              t3 = r46 * t1;
            }
            if (ik == 0) break;
            in_t1 = r23 * t2;
            in_a1 = (int)in_t1;
            in_a2 = t2 - t23 * in_a1;

            in_t1 = r23 * t2;
            in_x1 = (int)in_t1;
            in_x2 = t2 - t23 * in_x1;
            in_t1 = in_a1 * in_x2 + in_a2 * in_x1;
            in_t2 = (int)(r23 * in_t1);
            in_z = in_t1 - t23 * in_t2;
            in_t3 = t23 * in_z + in_a2 * in_x2;
            in_t4 = (int)(r46 * in_t3);
            t2 = in_t3 - t46 * in_t4;
            t3 = r46 * t2;
            kk = ik;
          }

          //--------------------------------------------------------------------
          //  Compute uniform pseudorandom numbers.
          //--------------------------------------------------------------------
          //vranlc(2 * NK, &t1, A, x);
          /*inline vranlc function*/
          in_t1 = r23 * A;
          in_a1 = (int)in_t1;
          in_a2 = A - t23 * in_a1;

          for(i=0; i<2*NK; i++)
          {
            in_t1 = r23 * t1;
            in_x1 = (int)in_t1;
            in_x2 = t1 - t23 * in_x1;
            in_t1 = in_a1 * in_x2 + in_a2 * in_x1;
            in_t2 = (int)(r23 * in_t1);
            in_z = in_t1 - t23 * in_t2;
            in_t3 = t23*in_z + in_a2 *in_x2;
            in_t4 = (int)(r46 * in_t3);
            t1 = in_t3 - t46 * in_t4;
            xx[i*blksize + (k-1)] = r46 * t1;
          }

          //--------------------------------------------------------------------
          //  Compute Gaussian deviates by acceptance-rejection method and
          //  tally counts in concentri//square annuli.  This loop is not
          //  vectorizable.
          //--------------------------------------------------------------------
          //if (timers_enabled) timer_start(1);

          tmp_sx = 0.0;
          tmp_sy = 0.0;

          for (i = 0; i < NK; i++) {
            x1 = 2.0 * xx[2*i*blksize + (k-1)] - 1.0;
            x2 = 2.0 * xx[(2*i+1)*blksize + (k-1)] - 1.0;
            t1 = x1 * x1 + x2 * x2;
            if (t1 <= 1.0) {
              t2   = sqrt(-2.0 * log(t1) / t1);
              t3   = (x1 * t2);
              t4   = (x2 * t2);
              l    = MAX(fabs(t3), fabs(t4));
              qq[l*blksize + (k-1)] += 1.0;
              tmp_sx   = tmp_sx + t3;
              tmp_sy   = tmp_sy + t4;
            }
          }

          sx += tmp_sx;
          sy += tmp_sy;

        }
      }/*end acc parallel*/

      //  printf("sx=%f,sy=%f\n", sx, sy);
#ifndef CRPL_COMP
#pragma acc parallel num_gangs(NQ) num_workers(4) vector_length(32) \
        present(q[0:NQ],qq[0:blksize*NQ])
#elif CRPL_COMP == 0
#pragma acc kernels present(q[0:NQ],qq[0:blksize*NQ])
#endif
      {
#pragma acc loop gang reduction(+:gc) independent
        for(i=0; i<NQ; i++)
        {
          double sum_qi = 0.0;
#pragma acc loop worker vector reduction(+:sum_qi) independent
          for(k=0; k<blksize; k++)
            sum_qi = sum_qi + qq[i*blksize + k];
          /*sum of each column of qq/q[i] */
          q[i] += sum_qi;
          /*final sum of q*/
          gc += sum_qi;
        }
      }
    }

  }/*end acc data*/

  timer_stop(0);
  tm = timer_read(0);

  nit = 0;
  verified = 1;
  if (M == 24) {
    sx_verify_value = -3.247834652034740e+3;
    sy_verify_value = -6.958407078382297e+3;
  } else if (M == 25) {
    sx_verify_value = -2.863319731645753e+3;
    sy_verify_value = -6.320053679109499e+3;
  } else if (M == 28) {
    sx_verify_value = -4.295875165629892e+3;
    sy_verify_value = -1.580732573678431e+4;
  } else if (M == 30) {
    sx_verify_value =  4.033815542441498e+4;
    sy_verify_value = -2.660669192809235e+4;
  } else if (M == 32) {
    sx_verify_value =  4.764367927995374e+4;
    sy_verify_value = -8.084072988043731e+4;
  } else if (M == 36) {
    sx_verify_value =  1.982481200946593e+5;
    sy_verify_value = -1.020596636361769e+5;
  } else if (M == 40) {
    sx_verify_value = -5.319717441530e+05;
    sy_verify_value = -3.688834557731e+05;
  } else {
    verified = 0;
  }

  if (verified) {
    sx_err = fabs((sx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((sy - sy_verify_value) / sy_verify_value);
    verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
  }

  Mops = pow(2.0, M+1) / tm / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", tm);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", gc);
  printf("Sums = %25.15lE %25.15lE\n", sx, sy);
  printf("Counts: \n");
  for (i = 0; i < NQ; i++) {
    printf("%3d%15.0lf\n", i, q[i]);
  }

  print_results("EP", CLASS, M+1, 0, 0, nit,
      tm, Mops, 
      "Random numbers generated",
      verified, NPBVERSION, COMPILETIME, CS1,
      CS2, CS3, CS4, CS5, CS6, CS7);

  if (timers_enabled) {
    if (tm <= 0.0) tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
    tt = timer_read(1);
    printf("Gaussian pairs: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
    tt = timer_read(2);
    printf("Random numbers: %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
  }

  free(xx);
  free(qq);

  return 0;
}
