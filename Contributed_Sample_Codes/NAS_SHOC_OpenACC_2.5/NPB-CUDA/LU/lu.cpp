//-------------------------------------------------------------------------//
//                                                                         //
//        N  A  S     P A R A L L E L     B E N C H M A R K S  3.3         //
//                                                                         //
//                         C U D A       V E R S I O N                     //
//                                                                         //
//                                   L U                                   //
//                                                                         //
//-------------------------------------------------------------------------//
//                                                                         //
//    This benchmark is a CUDA version of the NPB LU code.                 //
//    It is described in NAS Technical Report 99-011.                      //
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
// Authors of original FORTRAN version: S. Weeratunga, V. Venkatakrishnan  //
//                                      E. Barszcz, M. Yarrow, H. Jin      //
//                                                                         //
// CUDA implementation by: J. Duemmler                                     //
//                                                                         //
//-------------------------------------------------------------------------//

#include <stdio.h>
#include <math.h>
#include "main.h"

//---------------------------------------------------------------------
//
//   driver for the performance evaluation of the solver for
//   five coupled parabolic/elliptic partial differential equations.
//
//---------------------------------------------------------------------
int main (int argc, char **argv) {
	char benchclass = argc > 1 ? argv[1][0] : 'S';
	LU *lu = new LU();

	//---------------------------------------------------------------------
	//   read input data
	//---------------------------------------------------------------------
	lu->read_input(benchclass);

	//---------------------------------------------------------------------
	//   allocate CUDA device memory
	//---------------------------------------------------------------------
	lu->allocate_device_memory();

	//---------------------------------------------------------------------
	//   set up coefficients
	//---------------------------------------------------------------------
	lu->setcoeff();

	//---------------------------------------------------------------------
	//   set the boundary values for dependent variables
	//---------------------------------------------------------------------
	lu->setbv();

	//---------------------------------------------------------------------
	//   set the initial values for dependent variables
	//---------------------------------------------------------------------
	lu->setiv();

	//---------------------------------------------------------------------
	//   compute the forcing term based on prescribed exact solution
	//---------------------------------------------------------------------
	lu->erhs();

	//---------------------------------------------------------------------
	//   perform one SSOR iteration to touch all data pages
	//---------------------------------------------------------------------
	lu->ssor(1);

	//---------------------------------------------------------------------
	//   reset the boundary and initial values
	//---------------------------------------------------------------------
	lu->setbv();
	lu->setiv();

	//---------------------------------------------------------------------
	//   perform the SSOR iterations
	//---------------------------------------------------------------------
	lu->ssor(lu->get_itmax());

	//---------------------------------------------------------------------
	//   compute the solution error
	//---------------------------------------------------------------------
	lu->error();

	//---------------------------------------------------------------------
	//   compute the surface integral
	//---------------------------------------------------------------------
	lu->pintgr();

	//---------------------------------------------------------------------
	//   verification test
	//---------------------------------------------------------------------
	char verifyclass;
	bool verified = lu->verify(verifyclass);
	lu->print_results(verified, verifyclass);

	//---------------------------------------------------------------------
	//      More timers
	//---------------------------------------------------------------------
	lu->print_timers();

	delete lu;

	return EXIT_SUCCESS;
}

LU::LU() {
	timers = new Timers();
	get_cuda_info();
}

LU::~LU() {
	free_device_memory();
}


void LU::read_input(char benchclass) {
	//---------------------------------------------------------------------
	//    if input file does not exist, it uses defaults
	//       ipr = 1 for detailed progress output
	//       inorm = how often the norm is printed (once every inorm iterations)
	//       itmax = number of pseudo time steps
	//       dt = time step
	//       omega 1 over-relaxation factor for SSOR
	//       tolrsd = steady state residual tolerance levels
	//       nx, ny, nz = number of grid points in x, y, z directions
	//---------------------------------------------------------------------
	printf("\n\n NAS Parallel Benchmarks (NPB3.3-CUDA) - LU Benchmark\n\n");

	FILE *file = fopen("inputlu.data", "rt");
	if (file != 0L) {
		char line[1024];
		printf("Reading from input file inputlu.data\n");

		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i %i", &ipr, &inorm);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i", &itmax);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf", &dt);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf", &omega);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%lf %lf %lf %lf %lf", &tolrsd[0], &tolrsd[1], &tolrsd[2], &tolrsd[3], &tolrsd[4]);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		fgets(line, sizeof(line)-1, file);
		sscanf(line, "%i %i %i", &nx, &ny, &nz);
		fclose(file);
	} else {
		ipr = IPR_DEFAULT;
		omega = OMEGA_DEFAULT;

		int problem_size;
		switch (benchclass) {
			case 's':
			case 'S': problem_size = 12; dt = 0.5; itmax = 50; break;
			case 'w':
			case 'W': problem_size = 33; dt = 1.5e-3; itmax = 300; break;
			case 'a':
			case 'A': problem_size = 64; dt = 2.0; itmax = 250; break;
			case 'b':
			case 'B': problem_size = 102; dt = 2.0; itmax = 250; break;
			case 'c':
			case 'C': problem_size = 162; dt = 2.0; itmax = 250; break;
			case 'd':
			case 'D': problem_size = 408; dt = 1.0; itmax = 300; break;
			case 'e':
			case 'E': problem_size = 1020; dt = 0.5; itmax = 300; break;
			default: printf("setparams: Internal error: invalid class %c\n", benchclass); exit(EXIT_FAILURE);
		}
		nx = ny = nz = problem_size;
		inorm = itmax;

		tolrsd[0] = TOLRSD1_DEF;
		tolrsd[1] = TOLRSD2_DEF;
		tolrsd[2] = TOLRSD3_DEF;
		tolrsd[3] = TOLRSD4_DEF;
		tolrsd[4] = TOLRSD5_DEF;
	}

	//---------------------------------------------------------------------
	//   check problem size
	//---------------------------------------------------------------------
	if (nx < 4 || ny < 4 || nz < 4) {
		printf("     PROBLEM SIZE IS TOO SMALL - \n     SET EACH OF NX, NY AND NZ AT LEAST EQUAL TO 5\n");
		exit(EXIT_FAILURE);
	}

	printf(" Size: %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations:                  %5d\n", itmax);
	printf("\n");

	//---------------------------------------------------------------------
	// Setup info for timers
	//---------------------------------------------------------------------
	if ((file = fopen("timer.flag", "r")) != NULL) {
		Timers::init_timer();
		timeron = true;
		fclose(file);
	} else timeron = false;
}

//---------------------------------------------------------------------
//  verification routine                         
//---------------------------------------------------------------------
bool LU::verify(char &verifyclass) {
	bool verified = true;
	verifyclass = 'U';

	//---------------------------------------------------------------------
	//   tolerance level
	//---------------------------------------------------------------------
	double epsilon = 1.0e-08;

	double xcrref[5], xceref[5], xciref, dtref;
	double xcrdif[5], xcedif[5], xcidif;
	for (int m = 0; m < 5; m++) {
		xcrref[m] = xceref[m] = 1.0;
	}
	xciref = dtref = 1.0;

	if (nx == 12 && ny == 12 && nz == 12 && itmax == 50) {
		verifyclass = 'S';
		dtref = 5.0e-1;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (12X12X12) grid,
		//   after 50 time steps, with  DT = 5.0d-01
		//---------------------------------------------------------------------
		xcrref[0] = 1.6196343210976702e-02;
		xcrref[1] = 2.1976745164821318e-03;
		xcrref[2] = 1.5179927653399185e-03;
		xcrref[3] = 1.5029584435994323e-03;
		xcrref[4] = 3.4264073155896461e-02;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (12X12X12) grid,
		//   after 50 time steps, with  DT = 5.0d-01
		//---------------------------------------------------------------------
		xceref[0] = 6.4223319957960924e-04;
		xceref[1] = 8.4144342047347926e-05;
		xceref[2] = 5.8588269616485186e-05;
		xceref[3] = 5.8474222595157350e-05;
		xceref[4] = 1.3103347914111294e-03;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (12X12X12) grid,
		//   after 50 time steps, with DT = 5.0d-01
		//---------------------------------------------------------------------
		xciref = 7.8418928865937083e+00;
	} else if (nx == 33 && ny == 33 && nz == 33 && itmax == 300) {
		verifyclass = 'W';
		dtref = 1.5e-3;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (33x33x33) grid,
		//   after 300 time steps, with  DT = 1.5d-3
		//---------------------------------------------------------------------
		xcrref[0] =   0.1236511638192e+02;
		xcrref[1] =   0.1317228477799e+01;
		xcrref[2] =   0.2550120713095e+01;
		xcrref[3] =   0.2326187750252e+01;
		xcrref[4] =   0.2826799444189e+02;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (33X33X33) grid,
		//---------------------------------------------------------------------
		xceref[0] =   0.4867877144216e+00;
		xceref[1] =   0.5064652880982e-01;
		xceref[2] =   0.9281818101960e-01;
		xceref[3] =   0.8570126542733e-01;
		xceref[4] =   0.1084277417792e+01;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (33X33X33) grid,
		//   after 300 time steps, with  DT = 1.5d-3
		//---------------------------------------------------------------------
		xciref    =   0.1161399311023e+02;
	} else if (nx == 64 && ny == 64 && nz == 64 && itmax == 250) {
		verifyclass = 'A';
		dtref = 2.0e+0;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (64X64X64) grid,
		//   after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xcrref[0] = 7.7902107606689367e+02;
		xcrref[1] = 6.3402765259692870e+01;
		xcrref[2] = 1.9499249727292479e+02;
		xcrref[3] = 1.7845301160418537e+02;
		xcrref[4] = 1.8384760349464247e+03;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (64X64X64) grid,
		//   after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xceref[0] = 2.9964085685471943e+01;
		xceref[1] = 2.8194576365003349e+00;
		xceref[2] = 7.3473412698774742e+00;
		xceref[3] = 6.7139225687777051e+00;
		xceref[4] = 7.0715315688392578e+01;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (64X64X64) grid,
		//   after 250 time steps, with DT = 2.0d+00
		//---------------------------------------------------------------------
		xciref = 2.6030925604886277e+01;
	} else if (nx == 102 && ny == 102 && nz == 102 && itmax == 250) {
		verifyclass = 'B';
		dtref = 2.0e+0;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (102X102X102) grid,
		//   after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xcrref[0] = 3.5532672969982736e+03;
		xcrref[1] = 2.6214750795310692e+02;
		xcrref[2] = 8.8333721850952190e+02;
		xcrref[3] = 7.7812774739425265e+02;
		xcrref[4] = 7.3087969592545314e+03;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (102X102X102) 
		//   grid, after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xceref[0] = 1.1401176380212709e+02;
		xceref[1] = 8.1098963655421574e+00;
		xceref[2] = 2.8480597317698308e+01;
		xceref[3] = 2.5905394567832939e+01;
		xceref[4] = 2.6054907504857413e+02;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (102X102X102) grid,
		//   after 250 time steps, with DT = 2.0d+00
		//---------------------------------------------------------------------
		xciref = 4.7887162703308227e+01;
	} else if (nx == 162 && ny == 162 && nz == 162 && itmax == 250) {
		verifyclass = 'C';
		dtref = 2.0e+0;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (162X162X162) grid,
		//   after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xcrref[0] = 1.03766980323537846e+04;
		xcrref[1] = 8.92212458801008552e+02;
		xcrref[2] = 2.56238814582660871e+03;
		xcrref[3] = 2.19194343857831427e+03;
		xcrref[4] = 1.78078057261061185e+04;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (162X162X162) 
		//   grid, after 250 time steps, with  DT = 2.0d+00
		//---------------------------------------------------------------------
		xceref[0] = 2.15986399716949279e+02;
		xceref[1] = 1.55789559239863600e+01;
		xceref[2] = 5.41318863077207766e+01;
		xceref[3] = 4.82262643154045421e+01;
		xceref[4] = 4.55902910043250358e+02;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (162X162X162) grid,
		//   after 250 time steps, with DT = 2.0d+00
		//---------------------------------------------------------------------
		xciref = 6.66404553572181300e+01;
	} else if (nx == 408 && ny == 408 && nz == 408 && itmax == 300) {
		verifyclass = 'D';
		dtref = 1.0e+0;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (408X408X408) grid,
		//   after 300 time steps, with  DT = 1.0d+00
		//---------------------------------------------------------------------
		xcrref[0] = 0.4868417937025e+05;
		xcrref[1] = 0.4696371050071e+04;
		xcrref[2] = 0.1218114549776e+05;
		xcrref[3] = 0.1033801493461e+05;
		xcrref[4] = 0.7142398413817e+05;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (408X408X408) 
		//   grid, after 300 time steps, with  DT = 1.0d+00
		//---------------------------------------------------------------------
		xceref[0] = 0.3752393004482e+03;
		xceref[1] = 0.3084128893659e+02;
		xceref[2] = 0.9434276905469e+02;
		xceref[3] = 0.8230686681928e+02;
		xceref[4] = 0.7002620636210e+03;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (408X408X408) grid,
		//   after 300 time steps, with DT = 1.0d+00
		//---------------------------------------------------------------------
		xciref =    0.8334101392503e+02;
	} else if (nx == 1020 && ny == 1020 && nz == 1020 && itmax == 300) {
		verifyclass = 'E';
		dtref = 0.5e+0;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of residual, for the (1020X1020X1020) grid,
		//   after 300 time steps, with  DT = 0.5d+00
		//---------------------------------------------------------------------
		xcrref[0] = 0.2099641687874e+06;
		xcrref[1] = 0.2130403143165e+05;
		xcrref[2] = 0.5319228789371e+05;
		xcrref[3] = 0.4509761639833e+05;
		xcrref[4] = 0.2932360006590e+06;

		//---------------------------------------------------------------------
		//   Reference values of RMS-norms of solution error, for the (1020X1020X1020) 
		//   grid, after 300 time steps, with  DT = 0.5d+00
		//---------------------------------------------------------------------
		xceref[0] = 0.4800572578333e+03;
		xceref[1] = 0.4221993400184e+02;
		xceref[2] = 0.1210851906824e+03;
		xceref[3] = 0.1047888986770e+03;
		xceref[4] = 0.8363028257389e+03;

		//---------------------------------------------------------------------
		//   Reference value of surface integral, for the (1020X1020X1020) grid,
		//   after 300 time steps, with DT = 0.5d+00
		//---------------------------------------------------------------------
		xciref =    0.9512163272273e+02;
	} else verified = false;

	//---------------------------------------------------------------------
	//    verification test for residuals if gridsize is one of 
	//    the defined grid sizes above (class .ne. 'U')
	//---------------------------------------------------------------------

	//---------------------------------------------------------------------
	//    Compute the difference of solution values and the known reference values.
	//---------------------------------------------------------------------
	for (int m = 0; m < 5; m++) {
		xcrdif[m] = fabs((rsdnm[m]-xcrref[m])/xcrref[m]);
		xcedif[m] = fabs((errnm[m]-xceref[m])/xceref[m]);
	}
	xcidif = fabs((frc-xciref)/xciref);

	//---------------------------------------------------------------------
	//    Output the comparison of computed results to known cases.
	//---------------------------------------------------------------------
	if (verifyclass != 'U') {
		printf("\n Verification being performed for class %c\n", verifyclass);
		printf(" Accuracy setting for epsilon = %20.13E\n", epsilon);
		verified = fabs(dt-dtref) < epsilon;
		if (!verified) {
			verifyclass = 'U';
			printf(" DT does not match the reference value of %15.8E\n", dtref);
		}
	} else printf(" Unknown class\n");

	if (verifyclass != 'U') printf(" Comparison of RMS-norms of residual\n");
	else printf(" RMS-norms of residual\n");

	for (int m = 0; m < 5; m++) {
		if (verifyclass == 'U') printf("          %2d  %20.13E\n", m+1, rsdnm[m]);
		else if (xcrdif[m] <= epsilon) printf("          %2d  %20.13E%20.13E%20.13E\n", m+1, rsdnm[m], xcrref[m], xcrdif[m]);
		else {
			verified = false;
			printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n", m+1, rsdnm[m], xcrref[m], xcrdif[m]);
		}
	}

	if (verifyclass != 'U') printf(" Comparison of RMS-norms of solution error\n");
	else printf(" RMS-norms of solution error\n");

	for (int m = 0; m < 5; m++) {
		if (verifyclass == 'U') printf("          %2d  %20.13E\n", m+1, errnm[m]);
		else if (xcedif[m] <= epsilon) printf("          %2d  %20.13E%20.13E%20.13E\n", m+1, errnm[m], xceref[m], xcedif[m]);
		else {
			verified = false;
			printf(" FAILURE: %2d  %20.13E%20.13E%20.13E\n", m+1, errnm[m], xceref[m], xcedif[m]);
		}
	}

	if (verifyclass != 'U') printf(" Comparison of surface integral\n");
	else printf(" Surface integral\n");

	if (verifyclass == 'U') printf("              %20.13E\n", frc);
	else if (xcidif <= epsilon) printf("              %20.13E%20.13E%20.13E\n", frc, xciref, xcidif);
	else {
		verified = false;
		printf(" FAILURE:     %20.13E%20.13E%20.13E\n", frc, xciref, xcidif);
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

void LU::print_results(const bool verified, const char verifyclass) {

	printf("\n\n LU Benchmark Completed.\n");
	printf(" Class           =             %12c\n", verifyclass);
	printf(" Size            =           %4dx%4dx%4d\n", nx, ny, nz);
	printf(" Iterations      =             %12d\n", itmax);
	printf(" Time in seconds =             %12.2f\n", maxtime);

	double mflops = 0.0;
	if (maxtime != 0.0) mflops = (double)itmax*(1984.77*(double)nx*(double)ny*(double)nz-10923.3*((double)(nx+ny+nz)/3.0)*((double)(nx+ny+nz)/3.0)+27770.9*((double)(nx+ny+nz)/3.0)-144010.0)/(maxtime*1000000.);
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
