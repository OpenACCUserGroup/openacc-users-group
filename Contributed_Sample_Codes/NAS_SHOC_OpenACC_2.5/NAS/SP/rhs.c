//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is a serial C version of the NPB SP code. This C        //
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
////  The OpenACC C version of the NAS SP code is developed by the           //
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

#include <math.h>
#include "header.h"

void compute_rhs()
{
  int i, j, k, m;
  double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
  int gp0, gp1, gp2;

  gp0 = grid_points[0];
  gp1 = grid_points[1];
  gp2 = grid_points[2];

  //---------------------------------------------------------------------
  // compute the reciprocal of density, and the kinetic energy, 
  // and the speed of sound. 
  //---------------------------------------------------------------------
#pragma acc data present(rho_i,u,qs,square,speed,rhs,forcing,us,vs,ws) 
{
/*get the value of rho_i,qs,square,us,vs,ws,speed*/
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(0)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 0; k <= gp2-1; k++) {
#pragma acc loop worker
    for (j = 0; j <= gp1-1; j++) {
#pragma acc loop vector
      for (i = 0; i <= gp0-1; i++) {
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
        //-------------------------------------------------------------------
        // (don't need speed and ainx until the lhs computation)
        //-------------------------------------------------------------------
        aux = c1c2*rho_inv* (u[4][k][j][i] - square[k][j][i]);
        speed[k][j][i] = sqrt(aux);
      }
    }
  }


  //---------------------------------------------------------------------
  // copy the exact forcing term to the right hand side;  because 
  // this forcing term is known, we can store it on the whole grid
  // including the boundary                   
  //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(gp2) num_workers(8) vector_length(32) //async(1)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 0; k <= gp2-1; k++) {
#pragma acc loop worker
    for (j = 0; j <= gp1-1; j++) {
#pragma acc loop vector
      for (i = 0; i <= gp0-1; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = forcing[m][k][j][i];
        }
      }
    }
  }

  //---------------------------------------------------------------------
  // compute xi-direction fluxes 
  //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //wait(0, 1)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        uijk = us[k][j][i];
        up1  = us[k][j][i+1];
        um1  = us[k][j][i-1];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dx1tx1 * 
          (u[0][k][j][i+1] - 2.0*u[0][k][j][i] + u[0][k][j][i-1]) -
          tx2 * (u[1][k][j][i+1] - u[1][k][j][i-1]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dx2tx1 * 
          (u[1][k][j][i+1] - 2.0*u[1][k][j][i] + u[1][k][j][i-1]) +
          xxcon2*con43 * (up1 - 2.0*uijk + um1) -
          tx2 * (u[1][k][j][i+1]*up1 - u[1][k][j][i-1]*um1 +
                (u[4][k][j][i+1] - square[k][j][i+1] -
                 u[4][k][j][i-1] + square[k][j][i-1]) * c2);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dx3tx1 * 
          (u[2][k][j][i+1] - 2.0*u[2][k][j][i] + u[2][k][j][i-1]) +
          xxcon2 * (vs[k][j][i+1] - 2.0*vs[k][j][i] + vs[k][j][i-1]) -
          tx2 * (u[2][k][j][i+1]*up1 - u[2][k][j][i-1]*um1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dx4tx1 * 
          (u[3][k][j][i+1] - 2.0*u[3][k][j][i] + u[3][k][j][i-1]) +
          xxcon2 * (ws[k][j][i+1] - 2.0*ws[k][j][i] + ws[k][j][i-1]) -
          tx2 * (u[3][k][j][i+1]*up1 - u[3][k][j][i-1]*um1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dx5tx1 * 
          (u[4][k][j][i+1] - 2.0*u[4][k][j][i] + u[4][k][j][i-1]) +
          xxcon3 * (qs[k][j][i+1] - 2.0*qs[k][j][i] + qs[k][j][i-1]) +
          xxcon4 * (up1*up1 -       2.0*uijk*uijk + um1*um1) +
          xxcon5 * (u[4][k][j][i+1]*rho_i[k][j][i+1] - 
                2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j][i-1]*rho_i[k][j][i-1]) -
          tx2 * ( (c1*u[4][k][j][i+1] - c2*square[k][j][i+1])*up1 -
                  (c1*u[4][k][j][i-1] - c2*square[k][j][i-1])*um1 );
      }
    }

  } /*end k*/
  
    //---------------------------------------------------------------------
    // add fourth order xi-direction dissipation               
    //---------------------------------------------------------------------
  i = 1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(0)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++){
#pragma acc loop worker vector
    for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          (5.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
      }
    }
  }
      
  i = 2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(1)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++){
#pragma acc loop worker vector
    for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k][j][i-1] + 6.0*u[m][k][j][i] -
            4.0*u[m][k][j][i+1] + u[m][k][j][i+2]);
      }
    }
  }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(2)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++){
#pragma acc loop worker
	for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 3; i <= nx2-2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] + 
              u[m][k][j][i+2] );
        }
      }
    }
  }
    
 i = nx2-1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(3)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++){
#pragma acc loop worker vector
	for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k][j][i+1] );
      }
	}
  }
      
  i = nx2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async(4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++){
#pragma acc loop worker vector
	for (j = 1; j <= ny2; j++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j][i-2] - 4.0*u[m][k][j][i-1] + 5.0*u[m][k][j][i] );
      }
	}
  }

  //---------------------------------------------------------------------
  // compute eta-direction fluxes 
  //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //wait(0, 1, 2, 3, 4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        vijk = vs[k][j][i];
        vp1  = vs[k][j+1][i];
        vm1  = vs[k][j-1][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dy1ty1 * 
          (u[0][k][j+1][i] - 2.0*u[0][k][j][i] + u[0][k][j-1][i]) -
          ty2 * (u[2][k][j+1][i] - u[2][k][j-1][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dy2ty1 * 
          (u[1][k][j+1][i] - 2.0*u[1][k][j][i] + u[1][k][j-1][i]) +
          yycon2 * (us[k][j+1][i] - 2.0*us[k][j][i] + us[k][j-1][i]) -
          ty2 * (u[1][k][j+1][i]*vp1 - u[1][k][j-1][i]*vm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dy3ty1 * 
          (u[2][k][j+1][i] - 2.0*u[2][k][j][i] + u[2][k][j-1][i]) +
          yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
          ty2 * (u[2][k][j+1][i]*vp1 - u[2][k][j-1][i]*vm1 +
                (u[4][k][j+1][i] - square[k][j+1][i] - 
                 u[4][k][j-1][i] + square[k][j-1][i]) * c2);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dy4ty1 * 
          (u[3][k][j+1][i] - 2.0*u[3][k][j][i] + u[3][k][j-1][i]) +
          yycon2 * (ws[k][j+1][i] - 2.0*ws[k][j][i] + ws[k][j-1][i]) -
          ty2 * (u[3][k][j+1][i]*vp1 - u[3][k][j-1][i]*vm1);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dy5ty1 * 
          (u[4][k][j+1][i] - 2.0*u[4][k][j][i] + u[4][k][j-1][i]) +
          yycon3 * (qs[k][j+1][i] - 2.0*qs[k][j][i] + qs[k][j-1][i]) +
          yycon4 * (vp1*vp1       - 2.0*vijk*vijk + vm1*vm1) +
          yycon5 * (u[4][k][j+1][i]*rho_i[k][j+1][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k][j-1][i]*rho_i[k][j-1][i]) -
          ty2 * ((c1*u[4][k][j+1][i] - c2*square[k][j+1][i]) * vp1 -
                 (c1*u[4][k][j-1][i] - c2*square[k][j-1][i]) * vm1);
      }
    }

  }
    
	//---------------------------------------------------------------------
    // add fourth order eta-direction dissipation         
    //---------------------------------------------------------------------
  j = 1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async (0)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          ( 5.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
      }
    }
  }

  j = 2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async (1)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k][j-1][i] + 6.0*u[m][k][j][i] -
            4.0*u[m][k][j+1][i] + u[m][k][j+2][i]);
      }
    }
  }
    
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async (2)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker
	for (j = 3; j <= ny2-2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] + 
              u[m][k][j+2][i] );
        }
      }
    }
  }
    
  j = ny2-1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async (3)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k][j+1][i] );
      }
    }
  }
    
  j = ny2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //async (4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k][j-2][i] - 4.0*u[m][k][j-1][i] + 5.0*u[m][k][j][i] );
      }
    }
  }

  //---------------------------------------------------------------------
  // compute zeta-direction fluxes 
  //---------------------------------------------------------------------
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //wait(0, 1, 2, 3, 4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        wijk = ws[k][j][i];
        wp1  = ws[k+1][j][i];
        wm1  = ws[k-1][j][i];

        rhs[0][k][j][i] = rhs[0][k][j][i] + dz1tz1 * 
          (u[0][k+1][j][i] - 2.0*u[0][k][j][i] + u[0][k-1][j][i]) -
          tz2 * (u[3][k+1][j][i] - u[3][k-1][j][i]);

        rhs[1][k][j][i] = rhs[1][k][j][i] + dz2tz1 * 
          (u[1][k+1][j][i] - 2.0*u[1][k][j][i] + u[1][k-1][j][i]) +
          zzcon2 * (us[k+1][j][i] - 2.0*us[k][j][i] + us[k-1][j][i]) -
          tz2 * (u[1][k+1][j][i]*wp1 - u[1][k-1][j][i]*wm1);

        rhs[2][k][j][i] = rhs[2][k][j][i] + dz3tz1 * 
          (u[2][k+1][j][i] - 2.0*u[2][k][j][i] + u[2][k-1][j][i]) +
          zzcon2 * (vs[k+1][j][i] - 2.0*vs[k][j][i] + vs[k-1][j][i]) -
          tz2 * (u[2][k+1][j][i]*wp1 - u[2][k-1][j][i]*wm1);

        rhs[3][k][j][i] = rhs[3][k][j][i] + dz4tz1 * 
          (u[3][k+1][j][i] - 2.0*u[3][k][j][i] + u[3][k-1][j][i]) +
          zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
          tz2 * (u[3][k+1][j][i]*wp1 - u[3][k-1][j][i]*wm1 +
                (u[4][k+1][j][i] - square[k+1][j][i] - 
                 u[4][k-1][j][i] + square[k-1][j][i]) * c2);

        rhs[4][k][j][i] = rhs[4][k][j][i] + dz5tz1 * 
          (u[4][k+1][j][i] - 2.0*u[4][k][j][i] + u[4][k-1][j][i]) +
          zzcon3 * (qs[k+1][j][i] - 2.0*qs[k][j][i] + qs[k-1][j][i]) +
          zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
          zzcon5 * (u[4][k+1][j][i]*rho_i[k+1][j][i] - 
                  2.0*u[4][k][j][i]*rho_i[k][j][i] +
                    u[4][k-1][j][i]*rho_i[k-1][j][i]) -
          tz2 * ((c1*u[4][k+1][j][i] - c2*square[k+1][j][i])*wp1 -
                 (c1*u[4][k-1][j][i] - c2*square[k-1][j][i])*wm1);
      }
    }
  }

  //---------------------------------------------------------------------
  // add fourth order zeta-direction dissipation                
  //---------------------------------------------------------------------
  k = 1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(ny2) num_workers(8) vector_length(32) //async (0)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (j = 1; j <= ny2; j++) {
    #pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i]- dssp * 
          (5.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
      }
    }
  }

  k = 2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(ny2) num_workers(8) vector_length(32) //async (1)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (j = 1; j <= ny2; j++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
          (-4.0*u[m][k-1][j][i] + 6.0*u[m][k][j][i] -
            4.0*u[m][k+1][j][i] + u[m][k+2][j][i]);
      }
    }
  }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2-4) num_workers(8) vector_length(32) //async (2)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 3; k <= nz2-2; k++) {
#pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] - dssp * 
            ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
            6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] + 
              u[m][k+2][j][i] );
        }
      }
    }
  }

  k = nz2-1;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(ny2) num_workers(8) vector_length(32) //async (3)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (j = 1; j <= ny2; j++) {
#pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
     /*
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 
          6.0*u[m][k][j][i] - 4.0*u[m][k+1][j][i] );
      }
     */
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
    }
  }

  k = nz2;
#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(ny2) num_workers(8) vector_length(32) //async (4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (j = 1; j <= ny2; j++) {
    #pragma acc loop worker vector
    for (i = 1; i <= nx2; i++) {
    /*
      for (m = 0; m < 5; m++) {
        rhs[m][k][j][i] = rhs[m][k][j][i] - dssp *
          ( u[m][k-2][j][i] - 4.0*u[m][k-1][j][i] + 5.0*u[m][k][j][i] );
      }
    */
        rhs[0][k][j][i] = rhs[0][k][j][i] - dssp *
          ( u[0][k-2][j][i] - 4.0*u[0][k-1][j][i] + 5.0*u[0][k][j][i] );
        rhs[1][k][j][i] = rhs[1][k][j][i] - dssp *
          ( u[1][k-2][j][i] - 4.0*u[1][k-1][j][i] + 5.0*u[1][k][j][i] );
        rhs[2][k][j][i] = rhs[2][k][j][i] - dssp *
          ( u[2][k-2][j][i] - 4.0*u[2][k-1][j][i] + 5.0*u[2][k][j][i] );
        rhs[3][k][j][i] = rhs[3][k][j][i] - dssp *
          ( u[3][k-2][j][i] - 4.0*u[3][k-1][j][i] + 5.0*u[3][k][j][i] );
        rhs[4][k][j][i] = rhs[4][k][j][i] - dssp *
          ( u[4][k-2][j][i] - 4.0*u[4][k-1][j][i] + 5.0*u[4][k][j][i] );
    }
  }

#ifndef CRPL_COMP
#pragma acc parallel loop gang num_gangs(nz2) num_workers(8) vector_length(32) //wait(0, 1, 2, 3, 4)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang
#endif
  for (k = 1; k <= nz2; k++) {
#pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
#pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
        /*
        for (m = 0; m < 5; m++) {
          rhs[m][k][j][i] = rhs[m][k][j][i] * dt;
        }
        */
          rhs[0][k][j][i] = rhs[0][k][j][i] * dt;
          rhs[1][k][j][i] = rhs[1][k][j][i] * dt;
          rhs[2][k][j][i] = rhs[2][k][j][i] * dt;
          rhs[3][k][j][i] = rhs[3][k][j][i] * dt;
          rhs[4][k][j][i] = rhs[4][k][j][i] * dt;
      }
    }
  }
}/* end acc data */

}
