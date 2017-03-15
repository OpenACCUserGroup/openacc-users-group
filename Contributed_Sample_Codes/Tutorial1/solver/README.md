# Jacobi Iterative Method for solving a system of equations

See discussion for the Jacobi Method at
 https://en.wikipedia.org/wiki/Jacobi_method  (retrieved March 2017)

  We solve for vector x in Ax = b
    Rewrite the matrix A as a
       lower triangular (L),
       upper triangular (U),
    and diagonal matrix (D).

  Ax = (L + D + U)x = b

  rearrange to get: Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D

  we can do this iteratively: x_new = (b-(L+U)x_old)/D

These two programs (same program in C++ and Fortran)
iteratively solve the equation by computing xnew from
xold, until the difference between two consecutive
iterations has reached a tolerance.  The programs then
compute Ax and compare that against b as a correctness check.

make c++ to build and run the C++ version
make fortran to build and run the Fortran version

makefile default:
TIMER=/usr/bin/time	to measure execution time
ARGS=1000		size of the vector; additional arguments are maximum iterations and print frequency
OPT=			selected optimizations to, say, select target environment
NOPT=-fast -Minfo	normal optimizations, depends on the compilers used and target
CPP=pgc++		C++ compiler
FC=pgfortran		Fortran compiler
