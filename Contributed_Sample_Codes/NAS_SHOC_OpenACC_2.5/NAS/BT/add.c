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

//---------------------------------------------------------------------
// addition of update to the vector u//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;
  int gp22, gp12, gp02;

  gp22 = grid_points[2]-2;
  gp12 = grid_points[1]-2;
  gp02 = grid_points[0]-2;

#ifndef CRPL_COMP
#pragma acc parallel loop gang present(u,rhs) num_gangs(gp22) num_workers(4) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels loop gang present(u,rhs)
#endif
  for (k = 1; k <= gp22; k++) {
#pragma acc loop worker independent
    for (j = 1; j <= gp12; j++) {
#pragma acc loop vector independent
      for (i = 1; i <= gp02; i++) {
        /*
        for (m = 0; m < 5; m++) {
          u[m][k][j][i] = u[m][k][j][i] + rhs[m][k][j][i];
        }
         */
        u[0][k][j][i] = u[0][k][j][i] + rhs[0][k][j][i];
        u[1][k][j][i] = u[1][k][j][i] + rhs[1][k][j][i];
        u[2][k][j][i] = u[2][k][j][i] + rhs[2][k][j][i];
        u[3][k][j][i] = u[3][k][j][i] + rhs[3][k][j][i];
        u[4][k][j][i] = u[4][k][j][i] + rhs[4][k][j][i];
      }
    }
  }
}
