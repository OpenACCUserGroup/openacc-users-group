//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenMP C version of the NPB UA code. This OpenMP  //
//  C version is developed by the Center for Manycore Programming at Seoul //
//  National University and derived from the OpenMP Fortran versions in    //
//  "NPB3.3-OMP" developed by NAS.                                         //
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
//  Send comments or suggestions for this OpenMP C version to              //
//  cmp@aces.snu.ac.kr                                                     //
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

#include "header.h"

//---------------------------------------------------------------
// move element to proper location in morton space filling curve
//---------------------------------------------------------------
void move()
{
  int i, iside, jface, iel, ntemp, ii1, ii2, n1, n2, cb;

  n2 = 2*6*nelt;
  n1 = n2*2;

  nr_init_omp((int *)sje_new, n1, -1);
  nr_init_omp((int *)ijel_new, n2, -1);

  #pragma omp parallel default(shared) private(iel,i,iside,jface,cb,ntemp, \
                                               ii1,ii2) 
  {
  #pragma omp for
  for (iel = 0; iel < nelt; iel++) {
    i = mt_to_id[iel];
    treenew[iel] = tree[i];
    copy(xc_new[iel], xc[i], 8);
    copy(yc_new[iel], yc[i], 8);
    copy(zc_new[iel], zc[i], 8);

    for (iside = 0; iside < NSIDES; iside++) {
      jface = jjface[iside];
      cb = cbc[i][iside];
      xc_new[iel][iside] = xc[i][iside];
      yc_new[iel][iside] = yc[i][iside];
      zc_new[iel][iside] = zc[i][iside];
      cbc_new[iel][iside] = cb;

      if (cb == 2) {
        ntemp = sje[i][iside][0][0];
        ijel_new[iel][iside][0] = 0;
        ijel_new[iel][iside][1] = 0;
        sje_new[iel][iside][0][0] = id_to_mt[ntemp];

      } else if (cb == 1) {
        ntemp = sje[i][iside][0][0];
        ijel_new[iel][iside][0] = ijel[i][iside][0];
        ijel_new[iel][iside][1] = ijel[i][iside][1];
        sje_new[iel][iside][0][0] = id_to_mt[ntemp];

      } else if (cb == 3) {
        for (ii2 = 0; ii2 < 2; ii2++) {
          for (ii1 = 0; ii1 < 2; ii1++) {
            ntemp = sje[i][iside][ii2][ii1];
            ijel_new[iel][iside][0] = 0;
            ijel_new[iel][iside][1] = 0;
            sje_new[iel][iside][ii2][ii1] = id_to_mt[ntemp];
          }
        }

      } else if (cb == 0) {
        sje_new[iel][iside][0][0] = -1;
        sje_new[iel][iside][1][0] = -1;
        sje_new[iel][iside][0][1] = -1;
        sje_new[iel][iside][1][1] = -1;
      } 
    }

    copy(ta2[iel][0][0], ta1[i][0][0], NXYZ);
  }

  #pragma omp for
  for (iel = 0; iel < nelt; iel++) {
    copy(xc[iel], xc_new[iel], 8);
    copy(yc[iel], yc_new[iel], 8);
    copy(zc[iel], zc_new[iel], 8);
    copy(ta1[iel][0][0], ta2[iel][0][0], NXYZ);
    ncopy(sje[iel][0][0], sje_new[iel][0][0], 4*6);
    ncopy(ijel[iel][0], ijel_new[iel][0], 2*6);
    ncopy(cbc[iel], cbc_new[iel], 6);

    mt_to_id[iel] = iel;
    id_to_mt[iel] = iel;
    tree[iel] = treenew[iel];
  }
  } //end parallel
}
