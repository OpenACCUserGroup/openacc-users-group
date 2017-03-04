#ifndef _MAIN_H_
#define _MAIN_H_

#include <stdlib.h>

#define NPB_VERSION "3.3.1"

#define min(x,y) (x) <= (y) ? (x) : (y)
#define max(x,y) (x) >= (y) ? (x) : (y)

// block sizes for CUDA kernels
#define SOLVE_BLOCK 8
#define NORM_BLOCK 32
#define ERHS_BLOCK 32

// timer constants
#define t_total 0
#define t_rhsx 1
#define t_rhsy 2
#define t_rhsz 3
#define t_rhs 4
#define t_xsolve 5
#define t_ysolve 6
#define t_zsolve 7
#define t_rdis1 8
#define t_rdis2 9
#define t_add 10
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

class BT {
	int timeron;
	int niter;
	int nx, ny, nz;
	double dt;

	double *u, *forcing, *rhs, *lhs, *rho_i, *us, *vs, *ws, *qs, *square, *rmsbuf;
	double xce[5], xcr[5];

	char CUDAname[256];
	int CUDAmp, CUDAclock, CUDAmemclock, CUDAl2cache;
	size_t CUDAmem;

	Timers *timers;
	double tmax;
public:
	BT();
	~BT();

	void read_input(char benchclass);

	void allocate_device_memory();
	void free_device_memory();
	void get_cuda_info();

	void set_constants();
	void initialize();
	void exact_rhs();

	void adi(bool singlestep);
	void add();
	void compute_rhs();
	void x_solve();
	void y_solve();
	void z_solve();

	void error_norm();
	void rhs_norm();

	bool verify(char &verifyclass);
	void print_results(const bool verified, const char verifyclass);
	inline void print_timers() const { if (timeron) timers->timer_print(); }
};

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
#define dxmax max(dx3,dx4)
#define dymax max(dy2,dy4)
#define dzmax max(dz2,dz3)
//---------------------------------------------------------------------
//   fourth difference dissipation
//---------------------------------------------------------------------
#define dssp (( max(max(dx1, dy1), dz1) ) * .25)
#define c4dssp (4.0*dssp)
#define c5dssp (5.0*dssp)

#define c1 1.4
#define c2 0.4
#define c3 0.1
#define c4 1.0
#define c5 1.4
#define c1c2 (c1*c2)
#define c1c5 (c1*c5)
#define c3c4 (c3*c4)
#define c1345 (c1c5*c3c4)
#define conz1 (1.0-c1c5)
#define c2iv 2.5
#define con43 (4.0/3.0)
#define con16 (1.0/6.0)

// macros to linearize multidimensional array accesses 
#define u(m,i,j,k) u[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define rhs(m,i,j,k) rhs[m+(i)*5+(j)*5*nx+(k)*5*nx*ny]
#define forcing(m,i,j,k) forcing[(i)+nx*((j)+ny*((k)+nz*(m)))]
#define rho_i(i,j,k) rho_i[i+(j)*nx+(k)*nx*ny]
#define us(i,j,k) us[i+(j)*nx+(k)*nx*ny]
#define vs(i,j,k) vs[i+(j)*nx+(k)*nx*ny]
#define ws(i,j,k) ws[i+(j)*nx+(k)*nx*ny]
#define square(i,j,k) square[i+(j)*nx+(k)*nx*ny]
#define qs(i,j,k) qs[i+(j)*nx+(k)*nx*ny]
#define fjac(a,b,i) fjac[(a)+(b)*5+(i)*25]
#define njac(a,b,i) njac[(a)+(b)*5+(i)*25]

#endif
