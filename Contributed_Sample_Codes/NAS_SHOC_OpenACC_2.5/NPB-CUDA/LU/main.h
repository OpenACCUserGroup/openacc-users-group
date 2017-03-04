#ifndef _MAIN_H_
#define _MAIN_H_

#include <stdlib.h>

#define NPB_VERSION "3.3.1"

#define min(x,y) (x) <= (y) ? (x) : (y)
#define max(x,y) (x) >= (y) ? (x) : (y)

// block sizes for CUDA kernels
#define RHSX_BLOCK 32
#define RHSY_BLOCK 32
#define RHSZ_BLOCK 32
#define PINTGR_BLOCK 8
#define NORM_BLOCK 32

// timer constants
#define t_total 0
#define t_rhsx 1
#define t_rhsy 2
#define t_rhsz 3
#define t_rhs 4
#define t_jacld 5
#define t_blts 6
#define t_jacu 7
#define t_buts 8
#define t_add 9
#define t_l2norm 10
#define t_last 11

class Timers {
	double *elapsed, *start;
	static char *t_names[t_last];

	double elapsed_time();
public:
	Timers();
	~Timers();

	static void init_timer();

	void timer_clear(const int timer);
	void timer_clear_all();
	double timer_read(const int timer);
	void timer_start (const int timer);
	void timer_stop (const int timer);
	void timer_print();
};

class LU {
	int timeron;
	int inorm, ipr, itmax;
	int nx, ny, nz;
	double dt, omega, tolrsd[5];

	double *u, *rsd, *frct, *rho_i, *qs;
	double *dev_norm_buf;
	double rsdnm[5], errnm[5], frc;

	char CUDAname[256];
	int CUDAmp, CUDAclock, CUDAmemclock, CUDAl2cache;
	size_t CUDAmem;

	Timers *timers;
	double maxtime;
public:
	LU();
	~LU();

	void read_input(char benchclass);

	void allocate_device_memory();
	void free_device_memory();
	void get_cuda_info();

	void setcoeff();

	void ssor(int niter);
	void rhs();
	void l2norm (const double *v, double *sum);
	void error();
	void pintgr();
	void setbv();
	void setiv();
	void erhs();

	bool verify(char &verifyclass);
	void print_results(const bool verified, const char verifyclass);
	inline void print_timers() const { if (timeron) timers->timer_print(); }

	inline int get_itmax() const { return itmax; }
};

#define IPR_DEFAULT 1
#define OMEGA_DEFAULT 1.2
#define TOLRSD1_DEF 1.0e-08
#define TOLRSD2_DEF 1.0e-08
#define TOLRSD3_DEF 1.0e-08
#define TOLRSD4_DEF 1.0e-08
#define TOLRSD5_DEF 1.0e-08
//---------------------------------------------------------------------
// diffusion coefficients
//---------------------------------------------------------------------
#define dx1 0.75
#define dx2 0.75
#define dx3 0.75
#define dx4 0.75
#define dx5 0.75
#define dy1 0.75
#define dy2 0.75
#define dy3 0.75
#define dy4 0.75
#define dy5 0.75
#define dz1 1.00
#define dz2 1.00
#define dz3 1.00
#define dz4 1.00
#define dz5 1.00
//---------------------------------------------------------------------
//   fourth difference dissipation
//---------------------------------------------------------------------
#define dssp (( max(max(dx1, dy1), dz1) ) / 4.0)

#define c1 1.4
#define c2 0.4
#define c3 0.1
#define c4 1.0
#define c5 1.4

// macros to linearize multidimensional array accesses 
#define u(m,i,j,k) u[(m)+5*((i)+nx*((j)+ny*(k)))]
#define v(m,i,j,k) v[(m)+5*((i)+nx*((j)+ny*(k)))]
#define rsd(m,i,j,k) rsd[(m)+5*((i)+nx*((j)+ny*(k)))]
#define frct(m,i,j,k) frct[(m)+5*((i)+nx*((j)+ny*(k)))]
#define rho_i(i,j,k) rho_i[(i)+nx*((j)+ny*(k))]
#define qs(i,j,k) qs[(i)+nx*((j)+ny*(k))]

#endif
