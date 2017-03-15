# Compute PI

See discussion for Improved Say of Determining Pi at
 http://mb-soft.com/public3/pi.html  (retrieved March 2017)
If we integrate 1/(1+x^2) for x=0:1, we get pi/4

These two programs (same program in C++ and Fortran)
numerically estimate the integral by dividing the range 0:1
into nsteps sections.

make c++ to build and run the C++ version
make fortran to build and run the Fortran version

makefile default:
TIMER=/usr/bin/time	to measure execution time
STEPS=1000		number of sections to divide the region 0:1 into
OPT=			selected optimizations to, say, select target environment
NOPT=-fast -Minfo	normal optimizations, depends on the compilers used and target
CPP=pgc++		C++ compiler
FC=pgfortran		Fortran compiler
