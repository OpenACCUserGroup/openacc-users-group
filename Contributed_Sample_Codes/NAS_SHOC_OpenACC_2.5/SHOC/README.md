# SHOC_OpenACC_2.5

OpenACC 2.5 versions of the level 1 SHOC benchmarks (md, reduction, and stencil2d). The parent project is located at [https://github.com/spino327/NAS_SHOC_OpenACC_2.5](https://github.com/spino327/NAS_SHOC_OpenACC_2.5).

## Make individual benchmarks

You can pass to the make command the following variables. Alternatively, you can export the env.

* CC: compiler to use.  
* DEFINES: string to be passed to make (assuming you have a DEFINES variable within your makefile).  
* TA: architecture for OpenACC (as the -ta flag for pgi). E.g. `TA=multicore` or `TA=nvidia,cc35`.  
* EXTRA_CFLAGS: extra flags to pass to the c compiler.  
* EXTRA_CLINKFLAGS: extra flags to pass to linker.
* PXM: Program execution model to use. E.g. `PXM=acc`, `PXM=omp`

### SHOC
Compiling the OpenACC version for the multicore: `make -f Makefile.acc CC=pgc++ TA=multicore`.
Compiling the OpenACC version for the GPU: `make -f Makefile.acc CC=pgc++`.
Compiling the OpenMP version: `make -f Makefile.omp CC=pgc++`.

## Compiler definitions

We provide Makefile definitions that guide the compiling process. You can read and modify these build system. 
For SHOC, these files are at `crpl_conf/make.def` and `crpl_conf/make.common`.

## From original [repo](https://github.com/vetter/shoc)

The Scalable HeterOgeneous Computing (SHOC) benchmark suite is a
collection of benchmark programs testing the performance and
stability of systems using computing devices with non-traditional architectures
for general purpose computing. Its initial focus is on systems containing
Graphics Processing Units (GPUs) and multi-core processors, and on the
OpenCL programming standard. It can be used on clusters as well as individual
hosts.

Documentation on configuring, building, and running the SHOC benchmark
programs is contained in the SHOC user manual, in the doc subdirectory
of the SHOC source code tree.  The file INSTALL.txt contains a sketch of
those instructions for rapid installation.

Installation should be familiar to anyone who is experienced with configure
and make, see the config directory for some examples.  Also, if your
platform requires regenerating the configure script, see build-aux/bootstrap.sh
and the manual for more details.

