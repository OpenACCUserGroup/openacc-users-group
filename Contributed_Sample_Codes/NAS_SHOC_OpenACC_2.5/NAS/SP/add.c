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

#include "header.h"

//---------------------------------------------------------------------
// addition of update to the vector u
//---------------------------------------------------------------------
void add()
{
  int i, j, k, m;

//  if (timeron) timer_start(t_add);
#ifndef CRPL_COMP
#pragma acc parallel present(rhs,u) num_gangs(nz2) num_workers(8) vector_length(32)
#elif CRPL_COMP == 0
#pragma acc kernels present(rhs,u)
#endif
{
  #pragma acc loop gang
  for (k = 1; k <= nz2; k++) {
    #pragma acc loop worker
    for (j = 1; j <= ny2; j++) {
      #pragma acc loop vector
      for (i = 1; i <= nx2; i++) {
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
//  if (timeron) timer_stop(t_add);
}
