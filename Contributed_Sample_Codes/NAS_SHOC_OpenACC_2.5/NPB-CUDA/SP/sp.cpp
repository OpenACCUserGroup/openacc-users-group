//-------------------------------------------------------------------------//
//                                                                         //
//        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3         //
//                                                                         //
//                        C U D A       V E R S I O N                      //
//                                                                         //
//                                   S P                                   //
//                                                                         //
//-------------------------------------------------------------------------//
//                                                                         //
//    This benchmark is a serial version of the NPB SP code.               //
//    Refer to NAS Technical Reports 95-020 and 99-011 for details.        //
//                                                                         //
//    Permission to use, copy, distribute and modify this software         //
//    for any purpose with or without fee is hereby granted.  We           //
//    request, however, that all derived work reference the NAS            //
//    Parallel Benchmarks 3.3. This software is provided "as is"           //
//    without express or implied warranty.                                 //
//                                                                         //
//    Information on NPB 3.3, including the technical report, the          //
//    original specifications, source code, results and information        //
//    on how to submit new results, is available at:                       //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//    Send comments or suggestions to  npb@nas.nasa.gov                    //
//                                                                         //
//          NAS Parallel Benchmarks Group                                  //
//          NASA Ames Research Center                                      //
//          Mail Stop: T27A-1                                              //
//          Moffett Field, CA   94035-1000                                 //
//                                                                         //
//          E-mail:  npb@nas.nasa.gov                                      //
//          Fax:     (650) 604-3957                                        //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
//                                                                         //
// Authors of original FORTRAN version: R. Van der Wijngaart, W. Saphir,   //
//                                      H. Jin                             //
//                                                                         //
// CUDA implementation by: J. Duemmler                                     //
//                                                                         //
//-------------------------------------------------------------------------//

#include <stdio.h>
#include <math.h>
#include "main.h"

int main(int argc, char **argv) {
	char benchclass = argc > 1 ? argv[1][0] : 'S';
	SP *sp = new SP();

	//---------------------------------------------------------------------
	//   read input data
	//---------------------------------------------------------------------
	sp->read_input(benchclass);

	//---------------------------------------------------------------------
	//   allocate CUDA device memory
	//---------------------------------------------------------------------
	sp->allocate_device_memory();

	sp->set_constants();

	sp->exact_rhs();

	sp->initialize();

	//---------------------------------------------------------------------
	//      do one time step to touch all code, and reinitialize
	//---------------------------------------------------------------------
	sp->adi(true);
	sp->initialize();

	//---------------------------------------------------------------------
	//   main time stepping loop
	//---------------------------------------------------------------------
	sp->adi(false);

	//---------------------------------------------------------------------
	//   verification test
	//---------------------------------------------------------------------
	char verifyclass;
	bool verified = sp->verify(verifyclass);
	sp->print_results(verified, verifyclass);

	//---------------------------------------------------------------------
	//      More timers
	//---------------------------------------------------------------------
	sp->print_timers();

	delete sp;
	return EXIT_SUCCESS;
}

SP::SP() {
	timers = new Timers();
	get_cuda_info();
}

SP::~SP() {
	free_device_memory();
}

//---------------------------------------------------------------------
//      Read input file (if it exists), else take
//      defaults from parameters
//---------------------------------------------------------------------
void SP::read_input(char benchclass) {
	FILE *file;

	if ((file = fopen("timer.flag", "r")) != NULL) {
		Timers::init_timer();
		timeron = true;
		fclose(file);
	} else timeron = false;

	if ((file = fopen("inputsp.data", "rt")) != NULL) {
		char line[1024];
		printf(" Reading from input file inputsp.data\n");
		
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i", &niter);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf", &dt);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i %i %i", &nx, &ny, &nz);
		fclose(file);
	} else {
//		printf(" No input file inputsp.data. Using compiled defaults\n");
		int problem_size;
		switch (benchclass) {
			case 's':
			case 'S': problem_size = 12; dt = 0.015; niter = 100; break;
			case 'w':
			case 'W': problem_size = 36; dt = 0.0015; niter = 400; break;
			case 'a':
			case 'A': problem_size = 64; dt = 0.0015; niter = 400; break;
			case 'b':
			case 'B': problem_size = 102; dt = 0.001; niter = 400; break;
			case 'c':
			case 'C': problem_size = 162; dt = 0.00067; niter = 400; break;
			case 'd':
			case 'D': problem_size = 408; dt = 0.00030; niter = 500; break;
			case 'e':
			case 'E': problem_size = 1020; dt = 0.0001; niter = 500; break;
			default: printf("setparams: Internal error: invalid class %c\n", benchclass); exit(EXIT_FAILURE);
		}
		nx = ny = nz = problem_size;
	}

	printf("\n\n NAS Parallel Benchmarks (NPB3.3-CUDA) - SP Benchmark\n\n");
	printf(" Size: %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations: %4d    dt: %10.6F\n", niter, dt);
	printf("\n");
}

//---------------------------------------------------------------------
//  verification routine                         
//---------------------------------------------------------------------
bool SP::verify(char &verifyclass) {
	bool verified = true;
	verifyclass = 'U';

	//---------------------------------------------------------------------
	//   tolerance level
	//---------------------------------------------------------------------
	double epsilon = 1.0e-08;

	//---------------------------------------------------------------------
	//   compute the error norm and the residual norm, and exit if not printing
	//---------------------------------------------------------------------
	error_norm();
	compute_rhs();
	rhs_norm();
	for (int m = 0; m < 5; m++) xcr[m] = xcr[m] / dt;

	double xcrref[5], xceref[5], dtref;
	for (int m = 0; m < 5; m++) xcrref[m] = xceref[m] = 1.0;
	dtref = 1.0;

	if (nx == 12 && ny == 12 && nz == 12 && niter == 100) {
		//---------------------------------------------------------------------
		//    reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02
		//---------------------------------------------------------------------
		verifyclass = 'S';
		dtref = 1.5e-2;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 2.7470315451339479e-02;
		xcrref[1] = 1.0360746705285417e-02;
		xcrref[2] = 1.6235745065095532e-02;
		xcrref[3] = 1.5840557224455615e-02;
		xcrref[4] = 3.4849040609362460e-02;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 2.7289258557377227e-05;
		xceref[1] = 1.0364446640837285e-05;
		xceref[2] = 1.6154798287166471e-05;
		xceref[3] = 1.5750704994480102e-05;
		xceref[4] = 3.4177666183390531e-05;
	} else if (nx == 36 && ny == 36 && nz == 36 && niter == 400) {
		//---------------------------------------------------------------------
		//    reference data for 36X36X36 grids after 400 time steps, with DT = 1.5d-03
		//---------------------------------------------------------------------
		verifyclass = 'W';
		dtref = 1.5e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.1893253733584e-02;
		xcrref[1] = 0.1717075447775e-03;
		xcrref[2] = 0.2778153350936e-03;
		xcrref[3] = 0.2887475409984e-03;
		xcrref[4] = 0.3143611161242e-02;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.7542088599534e-04;
		xceref[1] = 0.6512852253086e-05;
		xceref[2] = 0.1049092285688e-04;
		xceref[3] = 0.1128838671535e-04;
		xceref[4] = 0.1212845639773e-03;
	} else if (nx == 64 && ny == 64 && nz == 64 && niter == 400) {
		//---------------------------------------------------------------------
		//    reference data for 64X64X64 grids after 400 time steps, with DT = 1.5d-03
		//---------------------------------------------------------------------
		verifyclass = 'A';
		dtref = 1.5e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 2.4799822399300195e0;
		xcrref[1] = 1.1276337964368832e0;
		xcrref[2] = 1.5028977888770491e0;
		xcrref[3] = 1.4217816211695179e0;
		xcrref[4] = 2.1292113035138280e0;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 1.0900140297820550e-04;
		xceref[1] = 3.7343951769282091e-05;
		xceref[2] = 5.0092785406541633e-05;
		xceref[3] = 4.7671093939528255e-05;
		xceref[4] = 1.3621613399213001e-04;
	} else if (nx == 102 && ny == 102 && nz == 102 && niter == 400) {
		//---------------------------------------------------------------------
		//    reference data for 102X102X102 grids after 400 time steps,
		//    with DT = 1.0d-03
		//---------------------------------------------------------------------
		verifyclass = 'B';
		dtref = 1.0e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.6903293579998e+02;
		xcrref[1] = 0.3095134488084e+02;
		xcrref[2] = 0.4103336647017e+02;
		xcrref[3] = 0.3864769009604e+02;
		xcrref[4] = 0.5643482272596e+02;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.9810006190188e-02;
		xceref[1] = 0.1022827905670e-02;
		xceref[2] = 0.1720597911692e-02;
		xceref[3] = 0.1694479428231e-02;
		xceref[4] = 0.1847456263981e-01;
	} else if (nx == 162 && ny == 162 && nz == 162 && niter == 400) {
		//---------------------------------------------------------------------
		//    reference data for 162X162X162 grids after 400 time steps,
		//    with DT = 0.67d-03
		//---------------------------------------------------------------------
		verifyclass = 'C';
		dtref = 0.67e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.5881691581829e+03;
		xcrref[1] = 0.2454417603569e+03;
		xcrref[2] = 0.3293829191851e+03;
		xcrref[3] = 0.3081924971891e+03;
		xcrref[4] = 0.4597223799176e+03;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.2598120500183e+00;
		xceref[1] = 0.2590888922315e-01;
		xceref[2] = 0.5132886416320e-01;
		xceref[3] = 0.4806073419454e-01;
		xceref[4] = 0.5483377491301e+00;
	} else if (nx == 408 && ny == 408 && nz == 408 && niter == 500) {
		//---------------------------------------------------------------------
		//    reference data for 408X408X408 grids after 500 time steps,
		//    with DT = 0.3d-03
		//---------------------------------------------------------------------
		verifyclass = 'D';
		dtref = 0.30e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.1044696216887e+05;
		xcrref[1] = 0.3204427762578e+04;
		xcrref[2] = 0.4648680733032e+04;
		xcrref[3] = 0.4238923283697e+04;
		xcrref[4] = 0.7588412036136e+04;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.5089471423669e+01;
		xceref[1] = 0.5323514855894e+00;
		xceref[2] = 0.1187051008971e+01;
		xceref[3] = 0.1083734951938e+01;
		xceref[4] = 0.1164108338568e+02;
	} else if (nx == 1020 && ny == 1020 && nz == 1020 && niter == 500) {
		//---------------------------------------------------------------------
		//    reference data for 1020X1020X1020 grids after 500 time steps,
		//    with DT = 0.1d-03
		//---------------------------------------------------------------------
		verifyclass = 'E';
		dtref = 0.10e-3;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.6255387422609e+05;
		xcrref[1] = 0.1495317020012e+05;
		xcrref[2] = 0.2347595750586e+05;
		xcrref[3] = 0.2091099783534e+05;
		xcrref[4] = 0.4770412841218e+05;

		//---------------------------------------------------------------------
		//    Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.6742735164909e+02;
		xceref[1] = 0.5390656036938e+01;
		xceref[2] = 0.1680647196477e+02;
		xceref[3] = 0.1536963126457e+02;
		xceref[4] = 0.1575330146156e+03;
	} else verified = false;

	//---------------------------------------------------------------------
	//    verification test for residuals if gridsize is one of 
	//    the defined grid sizes above (class .ne. 'U')
	//---------------------------------------------------------------------

	//---------------------------------------------------------------------
	//    Compute the difference of solution values and the known reference values.
	//---------------------------------------------------------------------
	double xcrdif[5], xcedif[5];
	for (int m = 0; m < 5; m++) {
		xcrdif[m] = fabs((xcr[m]-xcrref[m])/xcrref[m]);
		xcedif[m] = fabs((xce[m]-xceref[m])/xceref[m]);
	}

	//---------------------------------------------------------------------
	//    Output the comparison of computed results to known cases.
	//---------------------------------------------------------------------
	if (verifyclass != 'U') {
		printf(" Verification being performed for class %c\n", verifyclass);
		printf(" accuracy setting for epsilon = %20.13E\n", epsilon);
		verified = fabs(dt-dtref) <= epsilon;
		if (!verified) {
			verifyclass = 'U';
			printf(" DT does not match the reference value of %15.8E\n", dtref);
		}
	} else printf(" Unknown class\n");

	if (verifyclass != 'U') printf(" Comparison of RMS-norms of residual\n");
	else printf(" RMS-norms of residual\n");

	for (int m = 0; m < 5; m++) {
		if (verifyclass == 'U') printf("          %2d%20.13E\n", m+1, xcr[m]);
		else if (xcrdif[m] <= epsilon) printf("          %2d%20.13E%20.13E%20.13E\n", m+1, xcr[m], xcrref[m], xcrdif[m]);
		else {
			verified = false;
			printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n", m+1, xcr[m], xcrref[m], xcrdif[m]);
		}
	}

	if (verifyclass != 'U') printf(" Comparison of RMS-norms of solution error\n");
	else printf(" RMS-norms of solution error\n");

	for (int m = 0; m < 5; m++) {
		if (verifyclass == 'U') printf("          %2d%20.13E\n", m+1, xce[m]);
		else if (xcedif[m] <= epsilon) printf("          %2d%20.13E%20.13E%20.13E\n", m+1, xce[m], xceref[m], xcedif[m]);
		else {
			verified = false;
			printf(" FAILURE: %2d%20.13E%20.13E%20.13E\n", m+1, xce[m], xceref[m], xcedif[m]);
		}
	}

	if (verifyclass == 'U') {
		printf(" No reference values provided\n");
		printf(" No verification performed\n");
	} else {
		if (verified) printf(" Verification Successful\n");
		else printf(" Verification failed\n");
	}

	return verified;
}

void SP::print_results(const bool verified, const char verifyclass) {

	printf("\n\n SP Benchmark Completed.\n");
	printf(" Class           =             %12c\n", verifyclass);
	printf(" Size            =           %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations      =             %12d\n", niter);
	printf(" Time in seconds =             %12.2f\n", tmax);
	
	double mflops = 0.0;
	if (tmax != 0.0) {
		double n3 = nx*ny*nz;
		double t = (nx+ny+nz)/3.0;
		mflops = (881.174*n3-4683.91*t*t+11484.5*t-19272.4)*(double)niter / (tmax*1000000.0);
	}
	printf(" Mop/s total     =             %12.2f\n", mflops);
	printf(" Operation type  =           floating point\n");
	if (verified) printf(" Verification    =               SUCCESSFUL\n");
	else printf(" Verification    =             UNSUCCESSFUL\n");

	printf(" Version         =             %12s\n", NPB_VERSION);

	printf("\n");
	printf(" CUDA device     = %24s\n", CUDAname);
	printf(" GPU multiprocs  =             %12d\n", CUDAmp);
	printf(" GPU clock rate  =             %8.3f GHz\n", (double)CUDAclock/1000000.);
	printf(" GPU memory      =             %9.2f MB\n", (double)CUDAmem/(1024.*1024.));
	printf(" GPU mem clock   =             %8.3f GHz\n", (double)CUDAmemclock/1000000.);
	printf(" GPU L2 cache    =             %9.2f KB\n", (double)CUDAl2cache/1024.);

	printf("\n\n");

}
