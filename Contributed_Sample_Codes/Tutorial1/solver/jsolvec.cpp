// Jacobi iterative method for solving a system of linear equations
// This is guaranteed to converge if the matrix is diagonally dominant,
// so we artificially force the matrix to be diagonally dominant.
//  See https://en.wikipedia.org/wiki/Jacobi_method
//
//  We solve for vector x in Ax = b
//    Rewrite the matrix A as a
//       lower triangular (L),
//       upper triangular (U),
//    and diagonal matrix (D).
//
//  Ax = (L + D + U)x = b
//
//  rearrange to get: Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D
//
//  we can do this iteratively: x_new = (b-(L+U)x_old)/D

// build with TYPE=double (default) or TYPE=float
// build with TOLERANCE=0.001 (default) or TOLERANCE= any other value
// three arguments:
//   vector size
//   maximum iteration count
//   frequency of printing the residual (every n-th iteration)

#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>

using std::cout;

#ifndef TYPE
#define TYPE double
#endif

#define TOLERANCE 0.001

void
init_simple_diag_dom(int nsize, TYPE* A)
{
  int i, j;

  // In a diagonally-dominant matrix, the diagonal element
  // is greater than the sum of the other elements in the row.
  // Scale the matrix so the sum of the row elements is close to one.

  for (i = 0; i < nsize; ++i) {
    TYPE sum;
    sum = (TYPE)0;
    for (j = 0; j < nsize; ++j) {
      TYPE x;
      x = (rand() % 23) / (TYPE)1000;
      A[i*nsize + j] = x;
      sum += x;
    }
    // Fill diagonal element with the sum
    A[i*nsize + i] += sum;

    // scale the row so the final matrix is almost an identity matrix
    for (j = 0; j < nsize; j++)
      A[i*nsize + j] /= sum;
  }
} // init_simple_diag_dom

int
main(int argc, char **argv)
{
  int nsize; // A[nsize][nsize]
  int i, j, iters, max_iters, riter;
  double start_time, elapsed_time;
  TYPE residual, err, chksum;
  TYPE *A, *b, *x1, *x2, *xnew, *xold, *xtmp;

  // set matrix dimensions and allocate memory for matrices
  nsize = 0;
  if (argc > 1)
    nsize = atoi(argv[1]);
  if (nsize <= 0)
    nsize = 1000;

  max_iters = 0;
  if (argc > 2)
    max_iters = atoi(argv[2]);
  if (max_iters <= 0)
    max_iters = 5000;

  riter = 0;
  if (argc > 3)
    riter = atoi(argv[3]);
  if (riter <= 0)
    riter = 200;

  cout << "nsize = " << nsize << ", max_iters = " << max_iters << "\n";

  A = new TYPE[nsize*nsize];

  b = new TYPE[nsize];
  x1 = new TYPE[nsize];
  x2 = new TYPE[nsize];

  // generate a diagonally dominant matrix
  init_simple_diag_dom(nsize, A);

  // zero the x vectors, random values to the b vector
  for (i = 0; i < nsize; i++) {
    x1[i] = (TYPE)0.0;
    x2[i] = (TYPE)0.0;
    b[i] = (TYPE)(rand() % 51) / 100.0;
  }

  start_time = omp_get_wtime();

  //
  // jacobi iterative solver
  //

  residual = TOLERANCE + 1.0;
  iters = 0;
  xnew = x1;	// swap these pointers in each iteration
  xold = x2;
  while ((residual > TOLERANCE) && (iters < max_iters)) {
    ++iters;
    // swap input and output vectors
    xtmp = xnew;
    xnew = xold;
    xold = xtmp;

    for (i = 0; i < nsize; ++i) {
      TYPE rsum = (TYPE)0;
      for (j = 0; j < nsize; ++j) {
        if (i != j) rsum += A[i*nsize + j] * xold[j];
      }
      xnew[i] = (b[i] - rsum) / A[i*nsize + i];
    }
    //
    // test convergence, sqrt(sum((xnew-xold)**2))
    //
    residual = 0.0;
    for (i = 0; i < nsize; i++) {
      TYPE dif;
      dif = xnew[i] - xold[i];
      residual += dif * dif;
    }
    residual = sqrt((double)residual);
    if (iters % riter == 0 ) cout << "Iteration " << iters << ", residual is " << residual << "\n";
  }
  elapsed_time = omp_get_wtime() - start_time;
  cout << "\nConverged after " << iters << " iterations and " << elapsed_time << " seconds, residual is " << residual << "\n";

  //
  // test answer by multiplying my computed value of x by
  // the input A matrix and comparing the result with the
  // input b vector.
  //
  err = (TYPE)0.0;
  chksum = (TYPE)0.0;

  for (i = 0; i < nsize; i++) {
    TYPE tmp;
    xold[i] = (TYPE)0.0;
    for (j = 0; j < nsize; j++)
      xold[i] += A[i*nsize + j] * xnew[j];
    tmp = xold[i] - b[i];
    chksum += xnew[i];
    err += tmp * tmp;
  }
  err = sqrt((double)err);
  cout << "Solution error is " << err << "\n";
  if (err > TOLERANCE)
    cout << "****** Final Solution Out of Tolerance ******\n" << err << " > " << TOLERANCE << "\n";

  delete A;
  delete b;
  delete x1;
  delete x2;
  return 0;
}
