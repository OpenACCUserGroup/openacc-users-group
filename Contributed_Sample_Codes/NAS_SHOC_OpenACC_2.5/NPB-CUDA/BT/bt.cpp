//-------------------------------------------------------------------------//
//                                                                         //
//        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3         //
//                                                                         //
//                        C U D A       V E R S I O N                      //
//                                                                         //
//                                   B T                                   //
//                                                                         //
//-------------------------------------------------------------------------//
//                                                                         //
//    This benchmark is a serial version of the NPB BT code.               //
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
// Authors of original FORTRAN version: R. Van der Wijngaart, T. Harris,   //
//                                      M. Yarrow, H. Jin                  //
//                                                                         //
// CUDA implementation by: J. Duemmler                                     //
//                                                                         //
//-------------------------------------------------------------------------//

#include <stdio.h>
#include <math.h>
#include "main.h"

int main(int argc, char **argv) {
	char benchclass = argc > 1 ? argv[1][0] : 'S';
	BT *bt = new BT();

	//---------------------------------------------------------------------
	//      Root node reads input file (if it exists) else takes
	//      defaults from parameters
	//---------------------------------------------------------------------
	bt->read_input(benchclass);

	//---------------------------------------------------------------------
	//   allocate CUDA device memory
	//---------------------------------------------------------------------
	bt->allocate_device_memory();

	bt->set_constants();

	bt->initialize();

	bt->exact_rhs();

	//---------------------------------------------------------------------
	//      do one time step to touch all code, and reinitialize
	//---------------------------------------------------------------------
	bt->adi(true);
	bt->initialize();

	//---------------------------------------------------------------------
	//   main time stepping loop
	//---------------------------------------------------------------------
	bt->adi(false);

	//---------------------------------------------------------------------
	//   verification test
	//---------------------------------------------------------------------
	char verifyclass;
	bool verified = bt->verify(verifyclass);
	bt->print_results(verified, verifyclass);

	//---------------------------------------------------------------------
	//      More timers
	//---------------------------------------------------------------------
	bt->print_timers();

	delete bt;
	return EXIT_SUCCESS;
}

BT::BT() {
	timers = new Timers();
	get_cuda_info();
}

BT::~BT() {
	free_device_memory();
}

void BT::read_input(char benchclass) {
	FILE *file;

	if ((file = fopen("timer.flag", "r")) != NULL) {
		Timers::init_timer();
		timeron = true;
		fclose (file);
	} else timeron = false;

	if ((file = fopen("inputbt.data", "rt")) != NULL) {
		char line[1024];
		printf(" Reading from input file inputbt.data\n");
		
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i", &niter);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf", &dt);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i %i %i", &nx, &ny, &nz);
		fclose(file);
	} else {
//		printf(" No input file inputbt.data. Using compiled defaults\n");
		int problem_size;
		switch (benchclass) {
			case 's':
			case 'S': problem_size = 12; dt = 0.010; niter = 60; break;
			case 'w':
			case 'W': problem_size = 24; dt = 0.0008; niter = 200; break;
			case 'a':
			case 'A': problem_size = 64; dt = 0.0008; niter = 200; break;
			case 'b':
			case 'B': problem_size = 102; dt = 0.0003; niter = 200; break;
			case 'c':
			case 'C': problem_size = 162; dt = 0.0001; niter = 200; break;
			case 'd':
			case 'D': problem_size = 408; dt = 0.00002; niter = 250; break;
			case 'e':
			case 'E': problem_size = 1020; dt = 0.4e-5; niter = 250; break;
			default: printf("setparams: Internal error: invalid class %c\n", benchclass); exit(EXIT_FAILURE);
		}
		nx = ny = nz = problem_size;
	}

	printf("\n\n NAS Parallel Benchmarks (NPB3.3-CUDA) - BT Benchmark\n\n");
	printf(" Size: %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations: %4d    dt: %10.6F\n", niter, dt);
	printf("\n");
}

//---------------------------------------------------------------------
//  verification routine                         
//---------------------------------------------------------------------
bool BT::verify(char &verifyclass) {
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

	if (nx == 12 && ny == 12 && nz == 12 && niter == 60) {
		//---------------------------------------------------------------------
		//    reference data for 12X12X12 grids after 60 time steps, with DT = 1.0d-02
		//---------------------------------------------------------------------
		verifyclass = 'S';
		dtref = 1.0e-2;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 1.7034283709541311e-01;
		xcrref[1] = 1.2975252070034097e-02;
		xcrref[2] = 3.2527926989486055e-02;
		xcrref[3] = 2.6436421275166801e-02;
		xcrref[4] = 1.9211784131744430e-01;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 4.9976913345811579e-04;
		xceref[1] = 4.5195666782961927e-05;
		xceref[2] = 7.3973765172921357e-05;
		xceref[3] = 7.3821238632439731e-05;
		xceref[4] = 8.9269630987491446e-04;
	} else if (nx == 24 && ny == 24 && nz == 24 && niter == 200) {
		//---------------------------------------------------------------------
		//    reference data for 24X24X24 grids after 200 time steps, with DT = 0.8d-3
		//---------------------------------------------------------------------
		verifyclass = 'W';
		dtref = 0.8e-3;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.1125590409344e+03;
		xcrref[1] = 0.1180007595731e+02;
		xcrref[2] = 0.2710329767846e+02;
		xcrref[3] = 0.2469174937669e+02;
		xcrref[4] = 0.2638427874317e+03;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.4419655736008e+01;
		xceref[1] = 0.4638531260002e+00;
		xceref[2] = 0.1011551749967e+01;
		xceref[3] = 0.9235878729944e+00;
		xceref[4] = 0.1018045837718e+02;
	} else if (nx == 64 && ny == 64 && nz == 64 && niter == 200) {
		//---------------------------------------------------------------------
		//    reference data for 64X64X64 grids after 200 time steps, with DT = 0.8d-3
		//---------------------------------------------------------------------
		verifyclass = 'A';
		dtref = 0.8e-3;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 1.0806346714637264e+02;
		xcrref[1] = 1.1319730901220813e+01;
		xcrref[2] = 2.5974354511582465e+01;
		xcrref[3] = 2.3665622544678910e+01;
		xcrref[4] = 2.5278963211748344e+02;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 4.2348416040525025e+00;
		xceref[1] = 4.4390282496995698e-01;
		xceref[2] = 9.6692480136345650e-01;
		xceref[3] = 8.8302063039765474e-01;
		xceref[4] = 9.7379901770829278e+00;
	} else if (nx == 102 && ny == 102 && nz == 102 && niter == 200) {
		//---------------------------------------------------------------------
		//    reference data for 102X102X102 grids after 200 time steps,
		//    with DT = 3.0d-04
		//---------------------------------------------------------------------
		verifyclass = 'B';
		dtref = 3.0e-4;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 1.4233597229287254e+03;
		xcrref[1] = 9.9330522590150238e+01;
		xcrref[2] = 3.5646025644535285e+02;
		xcrref[3] = 3.2485447959084092e+02;
		xcrref[4] = 3.2707541254659363e+03;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 5.2969847140936856e+01;
		xceref[1] = 4.4632896115670668e+00;
		xceref[2] = 1.3122573342210174e+01;
		xceref[3] = 1.2006925323559144e+01;
		xceref[4] = 1.2459576151035986e+02;
	} else if (nx == 162 && ny == 162 && nz == 162 && niter == 200) {
		//---------------------------------------------------------------------
		//    reference data for 162X162X162 grids after 200 time steps,
		//    with DT = 1.0d-04
		//---------------------------------------------------------------------
		verifyclass = 'C';
		dtref = 1.0e-4;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.62398116551764615e+04;
		xcrref[1] = 0.50793239190423964e+03;
		xcrref[2] = 0.15423530093013596e+04;
		xcrref[3] = 0.13302387929291190e+04;
		xcrref[4] = 0.11604087428436455e+05;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.16462008369091265e+03;
		xceref[1] = 0.11497107903824313e+02;
		xceref[2] = 0.41207446207461508e+02;
		xceref[3] = 0.37087651059694167e+02;
		xceref[4] = 0.36211053051841265e+03;
	} else if (nx == 408 && ny == 408 && nz == 408 && niter == 250) {
		//---------------------------------------------------------------------
		//    reference data for 408x408x408 grids after 250 time steps,
		//    with DT = 0.2d-04
		//---------------------------------------------------------------------
		verifyclass = 'D';
		dtref = 0.2e-4;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.2533188551738e+05;
		xcrref[1] = 0.2346393716980e+04;
		xcrref[2] = 0.6294554366904e+04;
		xcrref[3] = 0.5352565376030e+04;
		xcrref[4] = 0.3905864038618e+05;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.3100009377557e+03;
		xceref[1] = 0.2424086324913e+02;
		xceref[2] = 0.7782212022645e+02;
		xceref[3] = 0.6835623860116e+02;
		xceref[4] = 0.6065737200368e+03;
	} else if (nx == 1020 && ny == 1020 && nz == 1020 && niter == 250) {
		//---------------------------------------------------------------------
		//    reference data for 1020x1020x1020 grids after 250 time steps,
		//    with DT = 0.4d-05
		//---------------------------------------------------------------------
		verifyclass = 'E';
		dtref = 0.4e-5;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of residual.
		//---------------------------------------------------------------------
		xcrref[0] = 0.9795372484517e+05;
		xcrref[1] = 0.9739814511521e+04;
		xcrref[2] = 0.2467606342965e+05;
		xcrref[3] = 0.2092419572860e+05;
		xcrref[4] = 0.1392138856939e+06;

		//---------------------------------------------------------------------
		//  Reference values of RMS-norms of solution error.
		//---------------------------------------------------------------------
		xceref[0] = 0.4327562208414e+03;
		xceref[1] = 0.3699051964887e+02;
		xceref[2] = 0.1089845040954e+03;
		xceref[3] = 0.9462517622043e+02;
		xceref[4] = 0.7765512765309e+03;
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

void BT::print_results(const bool verified, const char verifyclass) {

	printf("\n\n BT Benchmark Completed.\n");
	printf(" Class           =             %12c\n", verifyclass);
	printf(" Size            =           %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations      =             %12d\n", niter);
	printf(" Time in seconds =             %12.2f\n", tmax);
	
	double mflops = 0.0;
	if (tmax != 0.0) {
		double n3 = nx*ny*nz;
		double navg = (nx+ny+nz)/3.0;
		mflops = 1.0e-6*(double)niter*(3478.8*n3-17655.7*navg*navg+28023.7*navg)/tmax;
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
