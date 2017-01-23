PolyBench/ACC - OpenACC
=======================

##Contacts
* Tristan Vanderbruggen (tristan@udel.edu)
* William Killian (killian@udel.edu)

###OpenACC (RoseACC)

1. Set up `PATH` and `LD_LIBRARY_PATH` environment variables for RoseACC (see [RoseACC's Getting Started](https://github.com/tristanvdb/RoseACC-workspace))
2. Run `make exe` in target folder(s) with codes to generate executable(s)
3. Run the generated executable file(s).

##Usage

* Set environment variables PATH and LD_LIBRARY_PATH for RoseACC
* Run *make* (it will only build the parallelized benchmarks)

##Status

Benchmarks in bold have been parallelized. Benchmarks marked with [*] have reductions.

####datamining
* correlation
* covariance

####linear-algebra/kernels
* **2mm** [*]
* **3mm** [*]
* **atax** [*]
* **bicg** [*]
* cholesky
* doitgen
* gemm
* gemver
* gesummv
* mvt
* symm
* syr2k
* syrk
* trisolv
* trmm

####linear-algebra/solvers
* durbin
* dynprog
* gramschmidt
* lu
* ludcmp

####stencils
* adi
* convolution-2d
* convolution-3d
* fdtd-2d
* jacobi-1d-imper
* jacobi-2d-imper
* seidel-2d

