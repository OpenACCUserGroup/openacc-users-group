//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB BT code. This C        //
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
////  The OpenACC C version of the NAS BT code is developed by the           //
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

#include "header.h"
//#include "timers.h"
#include<stdio.h>

void compute_rhs()
{
  int i, j, k, m;
  double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;
  int gp01,gp11,gp21;
  int gp02,gp12,gp22;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];
  gp01 = grid_points[0]-1;
  gp11 = grid_points[1]-1;
  gp21 = grid_points[2]-1;
  gp02 = grid_points[0]-2;
  gp12 = grid_points[1]-2;
  gp22 = grid_points[2]-2;

  //  printf("gp01=%d, gp11=%d\n", gp01, gp11);
  //  printf("gp21=%d, gp02=%d\n", gp21, gp02);
  //  printf("gp12=%d, gp22=%d\n", gp12, gp22);

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound.
  //---------------------------------------------------------------------
#pragma acc data present(forcing,rho_i,u,us,vs,ws,square,qs,rhs) 
  {

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 0; k <= gp21; k++) {
#pragma acc loop worker independent
      for (j = 0; j <= gp11; j++) {
#pragma acc loop vector independent
        for (i = 0; i <= gp01; i++) {
          rho_inv = 1.0/u[0][k][j][i];
          rho_i[k][j][i] = rho_inv;
          us[k][j][i] = u[1][k][j][i] * rho_inv;
          vs[k][j][i] = u[2][k][j][i] * rho_inv;
          ws[k][j][i] = u[3][k][j][i] * rho_inv;
          square[k][j][i] = 0.5* (
              u[1][k][j][i]*u[1][k][j][i] +
              u[2][k][j][i]*u[2][k][j][i] +
              u[3][k][j][i]*u[3][k][j][i] ) * rho_inv;
          qs[k][j][i] = square[k][j][i] * rho_inv;
        }
      }
    }

    //---------------------------------------------------------------------
    // copy the exact forcingterm to the right hand side;  because
    // this forcingterm is known, we can store it on the whole grid
    // including the boundary
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc  parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 0; k <= gp21; k++) {
#pragma acc loop worker independent
      for (j = 0; j <= gp11; j++) {
#pragma acc loop vector independent
        for (i = 0; i <= gp01; i++) {
          rhs[0][k][j][i] = forcing[0][k][j][i];
          rhs[1][k][j][i] = forcing[1][k][j][i];
          rhs[2][k][j][i] = forcing[2][k][j][i];
          rhs[3][k][j][i] = forcing[3][k][j][i];
          rhs[4][k][j][i] = forcing[4][k][j][i];
        }
      }
    }

    //---------------------------------------------------------------------
    // compute xi-direction fluxes
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          uijk = us[k][j][i];
          up1  = us[k][j][i+1];
          um1  = us[k][j][i-1];

          rhs[0][k][j][i] = rhs[0][k][j][i] + dx1tx1 *
              (u[0][k][j][i+1] - 2.0*u[0][k][j][i] +
                  u[0][k][j][i-1]) -
                  tx2 * (u[1][k][j][i+1] - u[1][k][j][i-1]);

          rhs[1][k][j][i] = rhs[1][k][j][i] + dx2tx1 *
              (u[1][k][j][i+1] - 2.0*u[1][k][j][i] +
                  u[1][k][j][i-1]) +
                  xxcon2*con43 * (up1 - 2.0*uijk + um1) -
                  tx2 * (u[1][k][j][i+1]*up1 -
                      u[1][k][j][i-1]*um1 +
                      (u[4][k][j][i+1]- square[k][j][i+1]-
                          u[4][k][j][i-1]+ square[k][j][i-1])*
                          c2);

          rhs[2][k][j][i] = rhs[2][k][j][i] + dx3tx1 *
              (u[2][k][j][i+1] - 2.0*u[2][k][j][i] +
                  u[2][k][j][i-1]) +
                  xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] +
                      vs[k][j][i-1]) -
                      tx2 * (u[2][k][j][i+1]*up1 -
                          u[2][k][j][i-1]*um1);

          rhs[3][k][j][i] = rhs[3][k][j][i] + dx4tx1 *
              (u[3][k][j][i+1] - 2.0*u[3][k][j][i] +
                  u[3][k][j][i-1]) +
                  xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] +
                      ws[k][j][i-1]) -
                      tx2 * (u[3][k][j][i+1]*up1 -
                          u[3][k][j][i-1]*um1);

          rhs[4][k][j][i] = rhs[4][k][j][i] + dx5tx1 *
              (u[4][k][j][i+1] - 2.0*u[4][k][j][i] +
                  u[4][k][j][i-1]) +
                  xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] +
                      qs[k][j][i-1]) +
                      xxcon4 * (up1*up1 -       2.0*uijk*uijk +
                          um1*um1) +
                          xxcon5 * (u[4][k][j][i+1]*rho_i[k][j][i+1] -
                              2.0*u[4][k][j][i]*rho_i[k][j][i] +
                              u[4][k][j][i-1]*rho_i[k][j][i-1]) -
                              tx2 * ( (c1*u[4][k][j][i+1] -
                                  c2*square[k][j][i+1])*up1 -
                                  (c1*u[4][k][j][i-1] -
                                      c2*square[k][j][i-1])*um1 );
        }
      }
    }
    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker vector independent
      for (j = 1; j <= gp12; j++) {
        i = 1;
        rhs[0][k][j][i] = rhs[0][k][j][i]- dssp * 
            ( 5.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] +
                u[0][k][j][i+2]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
            ( 5.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] +
                u[1][k][j][i+2]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
            ( 5.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] +
                u[2][k][j][i+2]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
            ( 5.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] +
                u[3][k][j][i+2]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
            ( 5.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] +
                u[4][k][j][i+2]);

        i = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
            (-4.0*u[0][k][j][i-1] + 6.0*u[0][k][j][i] -
                4.0*u[0][k][j][i+1] + u[0][k][j][i+2]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
            (-4.0*u[1][k][j][i-1] + 6.0*u[1][k][j][i] -
                4.0*u[1][k][j][i+1] + u[1][k][j][i+2]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
            (-4.0*u[2][k][j][i-1] + 6.0*u[2][k][j][i] -
                4.0*u[2][k][j][i+1] + u[2][k][j][i+2]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
            (-4.0*u[3][k][j][i-1] + 6.0*u[3][k][j][i] -
                4.0*u[3][k][j][i+1] + u[3][k][j][i+2]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
            (-4.0*u[4][k][j][i-1] + 6.0*u[4][k][j][i] -
                4.0*u[4][k][j][i+1] + u[4][k][j][i+2]);
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 3; i <= gp02-2; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp*
              (  u[0][k][j][i-2] - 4.0*u[0][k][j][i-1] +
                  6.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] +
                  u[0][k][j][i+2] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp*
              (  u[1][k][j][i-2] - 4.0*u[1][k][j][i-1] +
                  6.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] +
                  u[1][k][j][i+2] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp*
              (  u[2][k][j][i-2] - 4.0*u[2][k][j][i-1] +
                  6.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] +
                  u[2][k][j][i+2] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp*
              (  u[3][k][j][i-2] - 4.0*u[3][k][j][i-1] +
                  6.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] +
                  u[3][k][j][i+2] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp*
              (  u[4][k][j][i-2] - 4.0*u[4][k][j][i-1] +
                  6.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] +
                  u[4][k][j][i+2] );
        }
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker vector independent
      for (j = 1; j <= gp12; j++) {
        i = gp0-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k][j][i-2] - 4.0*u[0][k][j][i-1] +
                6.0*u[0][k][j][i] - 4.0*u[0][k][j][i+1] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k][j][i-2] - 4.0*u[1][k][j][i-1] +
                6.0*u[1][k][j][i] - 4.0*u[1][k][j][i+1] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k][j][i-2] - 4.0*u[2][k][j][i-1] +
                6.0*u[2][k][j][i] - 4.0*u[2][k][j][i+1] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k][j][i-2] - 4.0*u[3][k][j][i-1] +
                6.0*u[3][k][j][i] - 4.0*u[3][k][j][i+1] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k][j][i-2] - 4.0*u[4][k][j][i-1] +
                6.0*u[4][k][j][i] - 4.0*u[4][k][j][i+1] );

        i = gp02;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k][j][i-2] - 4.*u[0][k][j][i-1] +
                5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k][j][i-2] - 4.*u[1][k][j][i-1] +
                5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k][j][i-2] - 4.*u[2][k][j][i-1] +
                5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k][j][i-2] - 4.*u[3][k][j][i-1] +
                5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k][j][i-2] - 4.*u[4][k][j][i-1] +
                5.*u[4][k][j][i] );
      }
    }

    //---------------------------------------------------------------------
    // compute eta-direction fluxes
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          vijk = vs[k][j][i];
          vp1  = vs[k][j+1][i];
          vm1  = vs[k][j-1][i];
          rhs[0][k][j][i] = rhs[0][k][j][i] + dy1ty1 *
              (u[0][k][j+1][i] - 2.0*u[0][k][j][i] +
                  u[0][k][j-1][i]) -
                  ty2 * (u[2][k][j+1][i] - u[2][k][j-1][i]);
          rhs[1][k][j][i] = rhs[1][k][j][i] + dy2ty1 *
              (u[1][k][j+1][i] - 2.0*u[1][k][j][i] +
                  u[1][k][j-1][i]) +
                  yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] +
                      us[k][j-1][i]) -
                      ty2 * (u[1][k][j+1][i]*vp1 -
                          u[1][k][j-1][i]*vm1);
          rhs[2][k][j][i] = rhs[2][k][j][i] + dy3ty1 *
              (u[2][k][j+1][i] - 2.0*u[2][k][j][i] +
                  u[2][k][j-1][i]) +
                  yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
                  ty2 * (u[2][k][j+1][i]*vp1 -
                      u[2][k][j-1][i]*vm1 +
                      (u[4][k][j+1][i] - square[k][j+1][i] -
                          u[4][k][j-1][i] + square[k][j-1][i])
                          *c2);
          rhs[3][k][j][i] = rhs[3][k][j][i] + dy4ty1 *
              (u[3][k][j+1][i] - 2.0*u[3][k][j][i] +
                  u[3][k][j-1][i]) +
                  yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] +
                      ws[k][j-1][i]) -
                      ty2 * (u[3][k][j+1][i]*vp1 -
                          u[3][k][j-1][i]*vm1);
          rhs[4][k][j][i] = rhs[4][k][j][i] + dy5ty1 *
              (u[4][k][j+1][i] - 2.0*u[4][k][j][i] +
                  u[4][k][j-1][i]) +
                  yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] +
                      qs[k][j-1][i]) +
                      yycon4 * (vp1*vp1       - 2.0*vijk*vijk +
                          vm1*vm1) +
                          yycon5 * (u[4][k][j+1][i]*rho_i[k][j+1][i] -
                              2.0*u[4][k][j][i]*rho_i[k][j][i] +
                              u[4][k][j-1][i]*rho_i[k][j-1][i]) -
                              ty2 * ((c1*u[4][k][j+1][i] -
                                  c2*square[k][j+1][i]) * vp1 -
                                  (c1*u[4][k][j-1][i] -
                                      c2*square[k][j-1][i]) * vm1);
        }
      }
    }
    //---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker vector independent
      for (i = 1; i <= gp02; i++) {
        j = 1;

        rhs[0][k][j][i] = rhs[0][k][j][i]- dssp *
            ( 5.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] +
                u[0][k][j+2][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
            ( 5.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] +
                u[1][k][j+2][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
            ( 5.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] +
                u[2][k][j+2][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
            ( 5.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] +
                u[3][k][j+2][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
            ( 5.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] +
                u[4][k][j+2][i]);

        j = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
            (-4.0*u[0][k][j-1][i] + 6.0*u[0][k][j][i] -
                4.0*u[0][k][j+1][i] + u[0][k][j+2][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
            (-4.0*u[1][k][j-1][i] + 6.0*u[1][k][j][i] -
                4.0*u[1][k][j+1][i] + u[1][k][j+2][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
            (-4.0*u[2][k][j-1][i] + 6.0*u[2][k][j][i] -
                4.0*u[2][k][j+1][i] + u[2][k][j+2][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
            (-4.0*u[3][k][j-1][i] + 6.0*u[3][k][j][i] -
                4.0*u[3][k][j+1][i] + u[3][k][j+2][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
            (-4.0*u[4][k][j-1][i] + 6.0*u[4][k][j][i] -
                4.0*u[4][k][j+1][i] + u[4][k][j+2][i]);
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 3; j <= gp1-4; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
              (  u[0][k][j-2][i] - 4.0*u[0][k][j-1][i] +
                  6.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] +
                  u[0][k][j+2][i] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
              (  u[1][k][j-2][i] - 4.0*u[1][k][j-1][i] +
                  6.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] +
                  u[1][k][j+2][i] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
              (  u[2][k][j-2][i] - 4.0*u[2][k][j-1][i] +
                  6.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] +
                  u[2][k][j+2][i] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
              (  u[3][k][j-2][i] - 4.0*u[3][k][j-1][i] +
                  6.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] +
                  u[3][k][j+2][i] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
              (  u[4][k][j-2][i] - 4.0*u[4][k][j-1][i] +
                  6.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] +
                  u[4][k][j+2][i] );
        }
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker vector independent
      for (i = 1; i <= gp02; i++) {
        j = gp1-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k][j-2][i] - 4.0*u[0][k][j-1][i] +
                6.0*u[0][k][j][i] - 4.0*u[0][k][j+1][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k][j-2][i] - 4.0*u[1][k][j-1][i] +
                6.0*u[1][k][j][i] - 4.0*u[1][k][j+1][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k][j-2][i] - 4.0*u[2][k][j-1][i] +
                6.0*u[2][k][j][i] - 4.0*u[2][k][j+1][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k][j-2][i] - 4.0*u[3][k][j-1][i] +
                6.0*u[3][k][j][i] - 4.0*u[3][k][j+1][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k][j-2][i] - 4.0*u[4][k][j-1][i] +
                6.0*u[4][k][j][i] - 4.0*u[4][k][j+1][i] );

        j = gp12;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k][j-2][i] - 4.*u[0][k][j-1][i] +
                5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k][j-2][i] - 4.*u[1][k][j-1][i] +
                5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k][j-2][i] - 4.*u[2][k][j-1][i] +
                5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k][j-2][i] - 4.*u[3][k][j-1][i] +
                5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k][j-2][i] - 4.*u[4][k][j-1][i] +
                5.*u[4][k][j][i] );
      }
    }

    //---------------------------------------------------------------------
    // compute zeta-direction fluxes
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          wijk = ws[k][j][i];
          wp1  = ws[k+1][j][i];
          wm1  = ws[k-1][j][i];

          rhs[0][k][j][i] = rhs[0][k][j][i] + dz1tz1 *
              (u[0][k+1][j][i] - 2.0*u[0][k][j][i] +
                  u[0][k-1][j][i]) -
                  tz2 * (u[3][k+1][j][i] - u[3][k-1][j][i]);
          rhs[1][k][j][i] = rhs[1][k][j][i] + dz2tz1 *
              (u[1][k+1][j][i] - 2.0*u[1][k][j][i] +
                  u[1][k-1][j][i]) +
                  zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] +
                      us[k-1][j][i]) -
                      tz2 * (u[1][k+1][j][i]*wp1 -
                          u[1][k-1][j][i]*wm1);
          rhs[2][k][j][i] = rhs[2][k][j][i] + dz3tz1 *
              (u[2][k+1][j][i] - 2.0*u[2][k][j][i] +
                  u[2][k-1][j][i]) +
                  zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] +
                      vs[k-1][j][i]) -
                      tz2 * (u[2][k+1][j][i]*wp1 -
                          u[2][k-1][j][i]*wm1);
          rhs[3][k][j][i] = rhs[3][k][j][i] + dz4tz1 *
              (u[3][k+1][j][i] - 2.0*u[3][k][j][i] +
                  u[3][k-1][j][i]) +
                  zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
                  tz2 * (u[3][k+1][j][i]*wp1 -
                      u[3][k-1][j][i]*wm1 +
                      (u[4][k+1][j][i] - square[k+1][j][i] -
                          u[4][k-1][j][i] + square[k-1][j][i])
                          *c2);
          rhs[4][k][j][i] = rhs[4][k][j][i] + dz5tz1 *
              (u[4][k+1][j][i] - 2.0*u[4][k][j][i] +
                  u[4][k-1][j][i]) +
                  zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] +
                      qs[k-1][j][i]) +
                      zzcon4 * (wp1*wp1 - 2.0*wijk*wijk +
                          wm1*wm1) +
                          zzcon5 * (u[4][k+1][j][i]*rho_i[k+1][j][i] -
                              2.0*u[4][k][j][i]*rho_i[k][j][i] +
                              u[4][k-1][j][i]*rho_i[k-1][j][i]) -
                              tz2 * ( (c1*u[4][k+1][j][i] -
                                  c2*square[k+1][j][i])*wp1 -
                                  (c1*u[4][k-1][j][i] -
                                      c2*square[k-1][j][i])*wm1);
        }
      }
    }
    //---------------------------------------------------------------------
    // add fourth order zeta-direction dissipation
    //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (j = 1; j <= gp12; j++) {
#pragma acc loop worker vector independent
      for (i = 1; i <= gp02; i++) {
        k = 1;
        rhs[0][k][j][i] = rhs[0][k][j][i]- dssp * 
            ( 5.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] +
                u[0][k+2][j][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i]- dssp * 
            ( 5.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] +
                u[1][k+2][j][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i]- dssp * 
            ( 5.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] +
                u[2][k+2][j][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i]- dssp * 
            ( 5.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] +
                u[3][k+2][j][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i]- dssp * 
            ( 5.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] +
                u[4][k+2][j][i]);

        k = 2;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
            (-4.0*u[0][k-1][j][i] + 6.0*u[0][k][j][i] -
                4.0*u[0][k+1][j][i] + u[0][k+2][j][i]);
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
            (-4.0*u[1][k-1][j][i] + 6.0*u[1][k][j][i] -
                4.0*u[1][k+1][j][i] + u[1][k+2][j][i]);
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
            (-4.0*u[2][k-1][j][i] + 6.0*u[2][k][j][i] -
                4.0*u[2][k+1][j][i] + u[2][k+2][j][i]);
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
            (-4.0*u[3][k-1][j][i] + 6.0*u[3][k][j][i] -
                4.0*u[3][k+1][j][i] + u[3][k+2][j][i]);
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
            (-4.0*u[4][k-1][j][i] + 6.0*u[4][k][j][i] -
                4.0*u[4][k+1][j][i] + u[4][k+2][j][i]);
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 3; k <= gp2-4; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] - dssp * 
              (  u[0][k-2][j][i] - 4.0*u[0][k-1][j][i] +
                  6.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] +
                  u[0][k+2][j][i] );
          rhs[1][k][j][i] = rhs[1][k][j][i] - dssp * 
              (  u[1][k-2][j][i] - 4.0*u[1][k-1][j][i] +
                  6.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] +
                  u[1][k+2][j][i] );
          rhs[2][k][j][i] = rhs[2][k][j][i] - dssp * 
              (  u[2][k-2][j][i] - 4.0*u[2][k-1][j][i] +
                  6.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] +
                  u[2][k+2][j][i] );
          rhs[3][k][j][i] = rhs[3][k][j][i] - dssp * 
              (  u[3][k-2][j][i] - 4.0*u[3][k-1][j][i] +
                  6.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] +
                  u[3][k+2][j][i] );
          rhs[4][k][j][i] = rhs[4][k][j][i] - dssp * 
              (  u[4][k-2][j][i] - 4.0*u[4][k-1][j][i] +
                  6.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] +
                  u[4][k+2][j][i] );
        }
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (j = 1; j <= gp12; j++) {
#pragma acc loop worker vector independent
      for (i = 1; i <= gp02; i++) {
        k = gp2-3;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k-2][j][i] - 4.0*u[0][k-1][j][i] +
                6.0*u[0][k][j][i] - 4.0*u[0][k+1][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k-2][j][i] - 4.0*u[1][k-1][j][i] +
                6.0*u[1][k][j][i] - 4.0*u[1][k+1][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k-2][j][i] - 4.0*u[2][k-1][j][i] +
                6.0*u[2][k][j][i] - 4.0*u[2][k+1][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k-2][j][i] - 4.0*u[3][k-1][j][i] +
                6.0*u[3][k][j][i] - 4.0*u[3][k+1][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k-2][j][i] - 4.0*u[4][k-1][j][i] +
                6.0*u[4][k][j][i] - 4.0*u[4][k+1][j][i] );

        k = gp22;
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
            ( u[0][k-2][j][i] - 4.*u[0][k-1][j][i] +
                5.*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
            ( u[1][k-2][j][i] - 4.*u[1][k-1][j][i] +
                5.*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
            ( u[2][k-2][j][i] - 4.*u[2][k-1][j][i] +
                5.*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
            ( u[3][k-2][j][i] - 4.*u[3][k-1][j][i] +
                5.*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
            ( u[4][k-2][j][i] - 4.*u[4][k-1][j][i] +
                5.*u[4][k][j][i] );
      }
    }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(192) num_workers(16) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
    for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
      for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
        for (i = 1; i <= gp02; i++) {
          rhs[0][k][j][i] = rhs[0][k][j][i] * dt;
          rhs[1][k][j][i] = rhs[1][k][j][i] * dt;
          rhs[2][k][j][i] = rhs[2][k][j][i] * dt;
          rhs[3][k][j][i] = rhs[3][k][j][i] * dt;
          rhs[4][k][j][i] = rhs[4][k][j][i] * dt;
        }
      }
    }

  }/*end acc data*/
}
