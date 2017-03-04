#include <stdio.h>
#include "main.h"

#define aa 0
#define bb 1
#define cc 2

namespace gpu_mod {
__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
__constant__ double dt, dtdssp;
__constant__ double xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1;
__constant__ double yycon1, yycon2, yycon3, yycon4, yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1;
__constant__ double zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1;
__constant__ double dnxm1, dnym1, dnzm1;
__constant__ double dttx1, dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1;
__constant__ double comz1, comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3;
__constant__ double ce[13][5];
}

static void inline HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define START_TIMER(timer) if (timeron) { HANDLE_ERROR(cudaDeviceSynchronize()); timers->timer_start(timer); }
#define STOP_TIMER(timer) if (timeron) { HANDLE_ERROR(cudaDeviceSynchronize()); timers->timer_stop(timer); }

void BT::adi (bool singlestep) {
	for (int i = 0; i < t_last; i++) timers->timer_clear(i);
	HANDLE_ERROR(cudaDeviceSynchronize());
	timers->timer_start(0);

	int itmax = singlestep ? 1 : niter;
	for (int step = 1; step <= itmax; step++) {
		if (step % 20 == 0 || step == 1 && !singlestep)
			printf(" Time step %4d\n", step);

		compute_rhs();
		x_solve();
		y_solve();
		z_solve();
		add();
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
	timers->timer_stop(0);
	tmax = timers->timer_read(0);
}

//---------------------------------------------------------------------
//     addition of update to the vector u
//---------------------------------------------------------------------
__global__ static void add_kernel (double *u, const double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k, m;

	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;
	m = threadIdx.y;

	u(m,i,j,k) += rhs(m,i,j,k);
}

void BT::add () {
	dim3 grid(ny-2,nz-2);
	dim3 block(nx-2,5);

	START_TIMER(t_add);
	add_kernel<<<grid,block>>>(u, rhs, nx, ny, nz);
	STOP_TIMER(t_add);
}

//---------------------------------------------------------------------
//      compute the reciprocal of density, and the kinetic energy
//---------------------------------------------------------------------
__global__ static void compute_rhs_kernel_1 (double *rho_i, double *us, double *vs, double *ws, double *qs, double *square, double *rhs, const double *forcing, const double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;

	double rho_inv = 1.0/u(0,i,j,k);
	rho_i(i,j,k) = rho_inv;
	us(i,j,k) = u(1,i,j,k) * rho_inv;
	vs(i,j,k) = u(2,i,j,k) * rho_inv;
	ws(i,j,k) = u(3,i,j,k) * rho_inv;
	square(i,j,k) = 0.5*(u(1,i,j,k)*u(1,i,j,k) + u(2,i,j,k)*u(2,i,j,k) + u(3,i,j,k)*u(3,i,j,k)) * rho_inv;
	qs(i,j,k) = square(i,j,k) * rho_inv;

	//---------------------------------------------------------------------
	// copy the exact forcing term to the right hand side;  because 
	// this forcing term is known, we can store it on the whole zone
	// including the boundary                   
	//---------------------------------------------------------------------
	for (m = 0; m < 5; m++) rhs(m,i,j,k) = forcing(m,i,j,k);
}

__global__ static void compute_rhs_kernel_2 (const double *rho_i, const double *us, const double *vs, const double *ws, const double *qs, const double *square, double *rhs, const double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rtmp[5];
	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;
	
	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//      compute xi-direction fluxes 
	//---------------------------------------------------------------------
	double uijk = us(i,j,k);
	double up1 = us(i+1,j,k);
	double um1 = us(i-1,j,k);
				
	rtmp[0] = rhs(0,i,j,k) + dx1tx1*(u(0,i+1,j,k) - 2.0*u(0,i,j,k) + u(0,i-1,j,k)) - tx2*(u(1,i+1,j,k)-u(1,i-1,j,k));
	rtmp[1] = rhs(1,i,j,k) + dx2tx1*(u(1,i+1,j,k) - 2.0*u(1,i,j,k) + u(1,i-1,j,k)) + xxcon2*con43*(up1-2.0*uijk+um1) - tx2*(u(1,i+1,j,k)*up1 - u(1,i-1,j,k)*um1 + (u(4,i+1,j,k)-square(i+1,j,k)-u(4,i-1,j,k)+square(i-1,j,k))*c2);
	rtmp[2] = rhs(2,i,j,k) + dx3tx1*(u(2,i+1,j,k) - 2.0*u(2,i,j,k) + u(2,i-1,j,k)) + xxcon2*(vs(i+1,j,k)-2.0*vs(i,j,k)+vs(i-1,j,k)) - tx2*(u(2,i+1,j,k)*up1 - u(2,i-1,j,k)*um1);
	rtmp[3] = rhs(3,i,j,k) + dx4tx1*(u(3,i+1,j,k) - 2.0*u(3,i,j,k) + u(3,i-1,j,k)) + xxcon2*(ws(i+1,j,k)-2.0*ws(i,j,k)+ws(i-1,j,k)) - tx2*(u(3,i+1,j,k)*up1 - u(3,i-1,j,k)*um1);
	rtmp[4] = rhs(4,i,j,k) + dx5tx1*(u(4,i+1,j,k) - 2.0*u(4,i,j,k) + u(4,i-1,j,k)) + xxcon3*(qs(i+1,j,k)-2.0*qs(i,j,k)+qs(i-1,j,k))+ xxcon4*(up1*up1-2.0*uijk*uijk+um1*um1) +
				xxcon5*(u(4,i+1,j,k)*rho_i(i+1,j,k) - 2.0*u(4,i,j,k)*rho_i(i,j,k) + u(4,i-1,j,k)*rho_i(i-1,j,k)) - tx2*((c1*u(4,i+1,j,k) - c2*square(i+1,j,k))*up1 - (c1*u(4,i-1,j,k) - c2*square(i-1,j,k))*um1 );
	//---------------------------------------------------------------------
	//      add fourth order xi-direction dissipation               
	//---------------------------------------------------------------------
	if (i == 1) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (5.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));
	} else if (i == 2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));
	} else if (i >= 3 && i < nx-3) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * ( u(m,i-2,j,k)-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k)+u(m,i+2,j,k));
	} else if (i == nx-3) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (u(m,i-2,j,k)-4.0*u(m,i-1,j,k)+6.0*u(m,i,j,k)-4.0*u(m,i+1,j,k) );
	} else if (i == nx-2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp * (u(m,i-2,j,k)-4.0*u(m,i-1,j,k) + 5.0*u(m,i,j,k));
	}

	//---------------------------------------------------------------------
	//      compute eta-direction fluxes 
	//---------------------------------------------------------------------
	double vijk = vs(i,j,k);
	double vp1 = vs(i,j+1,k);
	double vm1 = vs(i,j-1,k);
	rtmp[0] = rtmp[0] + dy1ty1*(u(0,i,j+1,k) - 2.0*u(0,i,j,k) + u(0,i,j-1,k)) - ty2*(u(2,i,j+1,k)-u(2,i,j-1,k));
	rtmp[1] = rtmp[1] + dy2ty1*(u(1,i,j+1,k) - 2.0*u(1,i,j,k) + u(1,i,j-1,k)) + yycon2*(us(i,j+1,k)-2.0*us(i,j,k)+us(i,j-1,k)) - ty2*(u(1,i,j+1,k)*vp1-u(1,i,j-1,k)*vm1);
	rtmp[2] = rtmp[2] + dy3ty1*(u(2,i,j+1,k) - 2.0*u(2,i,j,k) + u(2,i,j-1,k)) + yycon2*con43*(vp1-2.0*vijk+vm1) - ty2*(u(2,i,j+1,k)*vp1-u(2,i,j-1,k)*vm1+(u(4,i,j+1,k)-square(i,j+1,k)-u(4,i,j-1,k)+square(i,j-1,k))*c2);
	rtmp[3] = rtmp[3] + dy4ty1*(u(3,i,j+1,k) - 2.0*u(3,i,j,k) + u(3,i,j-1,k)) + yycon2*(ws(i,j+1,k)-2.0*ws(i,j,k)+ws(i,j-1,k))-ty2*(u(3,i,j+1,k)*vp1-u(3,i,j-1,k)*vm1);
	rtmp[4] = rtmp[4] + dy5ty1*(u(4,i,j+1,k) - 2.0*u(4,i,j,k) + u(4,i,j-1,k)) + yycon3*(qs(i,j+1,k)-2.0*qs(i,j,k)+qs(i,j-1,k)) + yycon4*(vp1*vp1-2.0*vijk*vijk+vm1*vm1) +
				yycon5*(u(4,i,j+1,k)*rho_i(i,j+1,k)-2.0*u(4,i,j,k)*rho_i(i,j,k)+u(4,i,j-1,k)*rho_i(i,j-1,k)) - ty2*((c1*u(4,i,j+1,k)-c2*square(i,j+1,k))*vp1 - (c1*u(4,i,j-1,k)-c2*square(i,j-1,k))*vm1);
	//---------------------------------------------------------------------
	//      add fourth order eta-direction dissipation         
	//---------------------------------------------------------------------
	if (j == 1) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(5.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));
	} else if (j == 2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));
	} else if (j >= 3 && j < ny-3) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k)+u(m,i,j+2,k));
	} else if (j == ny-3) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+6.0*u(m,i,j,k)-4.0*u(m,i,j+1,k));
	} else if (j == ny-2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j-2,k)-4.0*u(m,i,j-1,k)+5.0*u(m,i,j,k));
	}

	//---------------------------------------------------------------------
	//      compute zeta-direction fluxes 
	//---------------------------------------------------------------------
	double wijk = ws(i,j,k);
	double wp1 = ws(i,j,k+1);
	double wm1 = ws(i,j,k-1);

	rtmp[0] = rtmp[0] + dz1tz1*(u(0,i,j,k+1)-2.0*u(0,i,j,k)+u(0,i,j,k-1)) - tz2*(u(3,i,j,k+1)-u(3,i,j,k-1));
	rtmp[1] = rtmp[1] + dz2tz1*(u(1,i,j,k+1)-2.0*u(1,i,j,k)+u(1,i,j,k-1)) + zzcon2*(us(i,j,k+1)-2.0*us(i,j,k)+us(i,j,k-1)) - tz2*(u(1,i,j,k+1)*wp1-u(1,i,j,k-1)*wm1);
	rtmp[2] = rtmp[2] + dz3tz1*(u(2,i,j,k+1)-2.0*u(2,i,j,k)+u(2,i,j,k-1)) + zzcon2*(vs(i,j,k+1)-2.0*vs(i,j,k)+vs(i,j,k-1)) - tz2*(u(2,i,j,k+1)*wp1-u(2,i,j,k-1)*wm1);
	rtmp[3] = rtmp[3] + dz4tz1*(u(3,i,j,k+1)-2.0*u(3,i,j,k)+u(3,i,j,k-1)) + zzcon2*con43*(wp1-2.0*wijk+wm1) - tz2*(u(3,i,j,k+1)*wp1-u(3,i,j,k-1)*wm1+(u(4,i,j,k+1)-square(i,j,k+1)-u(4,i,j,k-1)+square(i,j,k-1))*c2);
	rtmp[4] = rtmp[4] + dz5tz1*(u(4,i,j,k+1)-2.0*u(4,i,j,k)+u(4,i,j,k-1)) + zzcon3*(qs(i,j,k+1)-2.0*qs(i,j,k)+qs(i,j,k-1)) + zzcon4*(wp1*wp1-2.0*wijk*wijk+wm1*wm1) +
				zzcon5*(u(4,i,j,k+1)*rho_i(i,j,k+1)-2.0*u(4,i,j,k)*rho_i(i,j,k)+u(4,i,j,k-1)*rho_i(i,j,k-1)) - tz2*((c1*u(4,i,j,k+1)-c2*square(i,j,k+1))*wp1-(c1*u(4,i,j,k-1)-c2*square(i,j,k-1))*wm1);
	//---------------------------------------------------------------------
	//      add fourth order zeta-direction dissipation                
	//---------------------------------------------------------------------
	if (k == 1) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(5.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
	} else if (k == 2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
	} else if (k >= 3 && k < nz-3) {
		for (m = 0; m < 5; m++)	rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
	} else if (k == nz-3) {
		for (m = 0; m < 5; m++)	rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1));
	} else if (k == nz-2) {
		for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+5.0*u(m,i,j,k));
	}

	for (m = 0; m < 5; m++) rhs(m,i,j,k) = rtmp[m] * dt;
}

void BT::compute_rhs () {
	dim3 grid1(ny,nz);
	dim3 grid2(ny-2,nz-2);

	START_TIMER(t_rhs);
	compute_rhs_kernel_1<<<grid1,nx>>>(rho_i, us, vs, ws, qs, square, rhs, forcing, u, nx, ny, nz);
	START_TIMER(t_rhsx);
	compute_rhs_kernel_2<<<grid2,nx-2>>>(rho_i, us, vs, ws, qs, square, rhs, u, nx, ny, nz);
	STOP_TIMER(t_rhsx);
	STOP_TIMER(t_rhs);
}

//---------------------------------------------------------------------
//     subtracts bvec=bvec - ablock*avec
//---------------------------------------------------------------------
__device__ static void matvec_sub_kernel (const int m, const double *ablock, const double *avec, double *bvec) {
	//---------------------------------------------------------------------
	//            rhs(i,ic,jc,kc) = rhs(i,ic,jc,kc) - lhs(i,1,ablock,ia)*
	//---------------------------------------------------------------------
	bvec[m] = bvec[m] - ablock[m+0*5]*avec[0] - ablock[m+1*5]*avec[1] - ablock[m+2*5]*avec[2] - ablock[m+3*5]*avec[3] - ablock[m+4*5]*avec[4];
}

//---------------------------------------------------------------------
// subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
//---------------------------------------------------------------------
__device__ static void matmul_sub_kernel (const int m, const double *ablock, const double *bblock, double *cblock) {
	cblock[m+0*5] = cblock[m+0*5] - ablock[m+0*5]*bblock[0+0*5] - ablock[m+1*5]*bblock[1+0*5] - ablock[m+2*5]*bblock[2+0*5] - ablock[m+3*5]*bblock[3+0*5] - ablock[m+4*5]*bblock[4+0*5];
	cblock[m+1*5] = cblock[m+1*5] - ablock[m+0*5]*bblock[0+1*5] - ablock[m+1*5]*bblock[1+1*5] - ablock[m+2*5]*bblock[2+1*5] - ablock[m+3*5]*bblock[3+1*5] - ablock[m+4*5]*bblock[4+1*5];
	cblock[m+2*5] = cblock[m+2*5] - ablock[m+0*5]*bblock[0+2*5] - ablock[m+1*5]*bblock[1+2*5] - ablock[m+2*5]*bblock[2+2*5] - ablock[m+3*5]*bblock[3+2*5] - ablock[m+4*5]*bblock[4+2*5];
	cblock[m+3*5] = cblock[m+3*5] - ablock[m+0*5]*bblock[0+3*5] - ablock[m+1*5]*bblock[1+3*5] - ablock[m+2*5]*bblock[2+3*5] - ablock[m+3*5]*bblock[3+3*5] - ablock[m+4*5]*bblock[4+3*5];
	cblock[m+4*5] = cblock[m+4*5] - ablock[m+0*5]*bblock[0+4*5] - ablock[m+1*5]*bblock[1+4*5] - ablock[m+2*5]*bblock[2+4*5] - ablock[m+3*5]*bblock[3+4*5] - ablock[m+4*5]*bblock[4+4*5];
}

__device__ static void binvcrhs_kernel (const int m, double *lhs, double *c, double *r) {
	double pivot;

	pivot = 1.0 / lhs[0+0*5];
	c[0+m*5] *= pivot;
	if (m > 0) lhs[0+m*5] *= pivot;
	else r[0] *= pivot;
	__syncthreads();

	c[1+m*5] -= lhs[1+0*5] * c[0+m*5];
	c[2+m*5] -= lhs[2+0*5] * c[0+m*5];
	c[3+m*5] -= lhs[3+0*5] * c[0+m*5];
	c[4+m*5] -= lhs[4+0*5] * c[0+m*5];
	if (m != 0) {
		lhs[m+1*5] -= lhs[m+0*5] * lhs[0+1*5];
		lhs[m+2*5] -= lhs[m+0*5] * lhs[0+2*5];
		lhs[m+3*5] -= lhs[m+0*5] * lhs[0+3*5];
		lhs[m+4*5] -= lhs[m+0*5] * lhs[0+4*5];
		r[m] -= lhs[m+0*5] * r[0];
	} 
	__syncthreads();

	pivot = 1.0/lhs[1+1*5];
	c[1+m*5] *= pivot;
	if (m > 1) lhs[1+m*5] *= pivot;
	else if (m == 0) r[1] *= pivot;
	__syncthreads();

	c[0+m*5] -= lhs[0+1*5] * c[1+m*5];
	c[2+m*5] -= lhs[2+1*5] * c[1+m*5];
	c[3+m*5] -= lhs[3+1*5] * c[1+m*5];
	c[4+m*5] -= lhs[4+1*5] * c[1+m*5];
	if (m != 1) {
		lhs[m+2*5] -= lhs[m+1*5] * lhs[1+2*5];
		lhs[m+3*5] -= lhs[m+1*5] * lhs[1+3*5];
		lhs[m+4*5] -= lhs[m+1*5] * lhs[1+4*5];
		r[m] -= lhs[m+1*5] * r[1];
	}
	__syncthreads();
	pivot = 1.0 / lhs[2+2*5];
	c[2+m*5] *= pivot;
	if (m > 2) lhs[2+m*5] *= pivot;
	else if (m == 0) r[2] *= pivot;
	__syncthreads();

	c[0+m*5] -= lhs[0+2*5] * c[2+m*5];
	c[1+m*5] -= lhs[1+2*5] * c[2+m*5];
	c[3+m*5] -= lhs[3+2*5] * c[2+m*5];
	c[4+m*5] -= lhs[4+2*5] * c[2+m*5];
	if (m != 2) {
		lhs[m+3*5] -= lhs[m+2*5] * lhs[2+3*5];
		lhs[m+4*5] -= lhs[m+2*5] * lhs[2+4*5];
		r[m] -= lhs[m+2*5] * r[2];
	}
	__syncthreads();

	pivot = 1.0/lhs[3+3*5];
	c[3+m*5] *= pivot;
	if (m == 4) lhs[3+4*5] *= pivot;
	else if (m == 0) r[3] *= pivot;
	__syncthreads();

	c[0+m*5] -= lhs[0+3*5] * c[3+m*5];
	c[1+m*5] -= lhs[1+3*5] * c[3+m*5];
	c[2+m*5] -= lhs[2+3*5] * c[3+m*5];
	c[4+m*5] -= lhs[4+3*5] * c[3+m*5];
	if (m != 3) {
		lhs[m+4*5] -= lhs[m+3*5] * lhs[3+4*5];
		r[m] -= lhs[m+3*5] * r[3];
	}
	__syncthreads();

	pivot = 1.0/lhs[4+4*5];
	c[4+m*5] *= pivot;
	if (m == 0) r[4] *= pivot;
	__syncthreads();

	c[0+m*5] -= lhs[0+4*5] * c[4+m*5];
	c[1+m*5] -= lhs[1+4*5] * c[4+m*5];
	c[2+m*5] -= lhs[2+4*5] * c[4+m*5];
	c[3+m*5] -= lhs[3+4*5] * c[4+m*5];
	if (m != 4) r[m] -= lhs[m+4*5] * r[4];
}

__device__ static void binvrhs_kernel (const int m, double *lhs, double *r) {
	double pivot;

	pivot = 1.0/lhs[0+0*5];
	if (m > 0) lhs[0+m*5] *= pivot;
	else r[0] *= pivot;
	__syncthreads();

	if (m != 0) {
		lhs[m+1*5] -= lhs[m+0*5] * lhs[0+1*5];
		lhs[m+2*5] -= lhs[m+0*5] * lhs[0+2*5];
		lhs[m+3*5] -= lhs[m+0*5] * lhs[0+3*5];
		lhs[m+4*5] -= lhs[m+0*5] * lhs[0+4*5];
		r[m] -= lhs[m+0*5] * r[0];
	}

	__syncthreads();
	pivot = 1.0/lhs[1+1*5];
	if (m > 1) lhs[1+m*5] *= pivot;
	else if (m == 0) r[1] *= pivot;
	__syncthreads();

	if (m != 1) {
		lhs[m+2*5] -= lhs[m+1*5] * lhs[1+2*5];
		lhs[m+3*5] -= lhs[m+1*5] * lhs[1+3*5];
		lhs[m+4*5] -= lhs[m+1*5] * lhs[1+4*5];
		r[m] -= lhs[m+1*5] * r[1];
	}

	__syncthreads();
	pivot = 1.0/lhs[2+2*5];
	if (m > 2) lhs[2+m*5] *= pivot;
	else if (m == 0) r[2] *= pivot;
	__syncthreads();

	if (m != 2) {
		lhs[m+3*5] -= lhs[m+2*5] * lhs[2+3*5];
		lhs[m+4*5] -= lhs[m+2*5] * lhs[2+4*5];
		r[m] -= lhs[m+2*5] * r[2];
	}

	__syncthreads();
	pivot = 1.0/lhs[3+3*5];
	if (m > 3) lhs[3+m*5] *= pivot;
	else if (m == 0) r[3] *= pivot;
	__syncthreads();

	if (m != 3) {
		lhs[m+4*5] -= lhs[m+3*5] * lhs[3+4*5];
		r[m] -= lhs[m+3*5] * r[3];
	}
	
	__syncthreads();
	if (m == 0) {
		pivot = 1.0/lhs[4+4*5];
		r[4] *= pivot;
	}
	__syncthreads();
	if (m != 4) r[m] -= lhs[m+4*5] * r[4];
}

//---------------------------------------------------------------------
//     
//     Performs line solves in X direction by first factoring
//     the block-tridiagonal matrix into an upper triangular matrix, 
//     and then performing back substitution to solve for the unknow
//     vectors of each line.  
//     
//     Make sure we treat elements zero to cell_size in the direction
//     of the sweep.
//     
//---------------------------------------------------------------------
#define lhs(a,b,c,i) lhs[(a)+5*((i)+nx*((b)+5*(c)))]
__global__ static void x_solve_kernel_1 (const double *rho_i, const double *qs, const double *square, const double *u, double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int j, k, jacofs, lhsofs, jacofs2;
	double tmp1, tmp2, tmp3, utmp[5];
	k = blockIdx.x+1;
	j = blockIdx.y+1;
	__shared__ double fjac[2*5*5*SOLVE_BLOCK];
	__shared__ double njac[2*5*5*SOLVE_BLOCK];
	lhs += (k-1+(j-1)*nz)*5*5*3*nx;

	using namespace gpu_mod;

	lhsofs = threadIdx.x-1;
	jacofs = threadIdx.x;
	while (lhsofs < nx-1) {
		//---------------------------------------------------------------------
		//     This function computes the left hand side in the xi-direction
		//     isize = nx-1
		//---------------------------------------------------------------------
		//---------------------------------------------------------------------
		//     determine a (labeled f) and n jacobians
		//---------------------------------------------------------------------
		jacofs2 = lhsofs+1;
		tmp1 = rho_i(jacofs2,j,k);
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		utmp[0] = u(0,jacofs2,j,k);
		utmp[1] = u(1,jacofs2,j,k);
		utmp[2] = u(2,jacofs2,j,k);
		utmp[3] = u(3,jacofs2,j,k);
		utmp[4] = u(4,jacofs2,j,k);
		//---------------------------------------------------------------------
		//     
		//---------------------------------------------------------------------
		fjac(0,0,jacofs) = 0.0;
		fjac(0,1,jacofs) = 1.0;
		fjac(0,2,jacofs) = 0.0;
		fjac(0,3,jacofs) = 0.0;
		fjac(0,4,jacofs) = 0.0;

		fjac(1,0,jacofs) = -(utmp[1] * tmp2 * utmp[1]) + c2*qs(jacofs2,j,k);
		fjac(1,1,jacofs) = (2.0-c2) *(utmp[1]/utmp[0]);
		fjac(1,2,jacofs) = -c2 * (utmp[2] * tmp1);
		fjac(1,3,jacofs) = -c2 * (utmp[3] * tmp1);
		fjac(1,4,jacofs) = c2;

		fjac(2,0,jacofs) = -(utmp[1]*utmp[2]) * tmp2;
		fjac(2,1,jacofs) = utmp[2] * tmp1;
		fjac(2,2,jacofs) = utmp[1] * tmp1;
		fjac(2,3,jacofs) = 0.0;
		fjac(2,4,jacofs) = 0.0;

		fjac(3,0,jacofs) = -(utmp[1]*utmp[3]) * tmp2;
		fjac(3,1,jacofs) = utmp[3] * tmp1;
		fjac(3,2,jacofs) = 0.0;
		fjac(3,3,jacofs) = utmp[1] * tmp1;
		fjac(3,4,jacofs) = 0.0;

		fjac(4,0,jacofs) = (c2*2.0*square(jacofs2,j,k) - c1*utmp[4]) * (utmp[1]*tmp2);
		fjac(4,1,jacofs) = c1*utmp[4]*tmp1 - c2*(utmp[1]*utmp[1]*tmp2 + qs(jacofs2,j,k));
		fjac(4,2,jacofs) = -c2 * (utmp[2]*utmp[1]) * tmp2;
		fjac(4,3,jacofs) = -c2 * (utmp[3]*utmp[1]) * tmp2;
		fjac(4,4,jacofs) = c1 * (utmp[1] * tmp1);

		njac(0,0,jacofs) = 0.0;
		njac(0,1,jacofs) = 0.0;
		njac(0,2,jacofs) = 0.0;
		njac(0,3,jacofs) = 0.0;
		njac(0,4,jacofs) = 0.0;

		njac(1,0,jacofs) = -con43 * c3c4 * tmp2 * utmp[1];
		njac(1,1,jacofs) = con43 * c3c4 * tmp1;
		njac(1,2,jacofs) = 0.0;
		njac(1,3,jacofs) = 0.0;
		njac(1,4,jacofs) = 0.0;

		njac(2,0,jacofs) = -c3c4 * tmp2 * utmp[2];
		njac(2,1,jacofs) = 0.0;
		njac(2,2,jacofs) = c3c4 * tmp1;
		njac(2,3,jacofs) = 0.0;
		njac(2,4,jacofs) = 0.0;

		njac(3,0,jacofs) = -c3c4 * tmp2 * utmp[3];
		njac(3,1,jacofs) = 0.0;
		njac(3,2,jacofs) = 0.0;
		njac(3,3,jacofs) = c3c4 * tmp1;
		njac(3,4,jacofs) = 0.0;

		njac(4,0,jacofs) = -(con43*c3c4-c1345)*tmp3*(utmp[1]*utmp[1]) - (c3c4-c1345)*tmp3*(utmp[2]*utmp[2]) - (c3c4-c1345)*tmp3*(utmp[3]*utmp[3]) - c1345*tmp2*utmp[4];
		njac(4,1,jacofs) = (con43*c3c4-c1345)*tmp2*utmp[1];
		njac(4,2,jacofs) = (c3c4-c1345)*tmp2*utmp[2];
		njac(4,3,jacofs) = (c3c4-c1345)*tmp2*utmp[3];
		njac(4,4,jacofs) = c1345*tmp1;
	
		//---------------------------------------------------------------------
		//     now jacobians set, so form left hand side in x direction
		//---------------------------------------------------------------------
		__syncthreads();
		if (lhsofs < 1) {
			jacofs2 = lhsofs == 0 ? 0 : nx-1;
			for (int m = 0; m < 5; m++) {
				for (int n = 0; n < 5; n++) {
					lhs(m,n,aa,jacofs2) = 0.0;
					lhs(m,n,bb,jacofs2) = 0.0;
					lhs(m,n,cc,jacofs2) = 0.0;
				}
				lhs(m,m,bb,jacofs2) = 1.0;
			}
		} else {
			tmp1 = dt * tx1;
			tmp2 = dt * tx2;

			jacofs2 = (2*SOLVE_BLOCK+jacofs-2) % (2*SOLVE_BLOCK);
			lhs(0,0,aa,lhsofs) = -tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dx1;
			lhs(0,1,aa,lhsofs) = -tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,aa,lhsofs) = -tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,aa,lhsofs) = -tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,aa,lhsofs) = -tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,aa,lhsofs) = -tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,aa,lhsofs) = -tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dx2;
			lhs(1,2,aa,lhsofs) = -tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,aa,lhsofs) = -tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,aa,lhsofs) = -tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);

			lhs(2,0,aa,lhsofs) = -tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,aa,lhsofs) = -tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,aa,lhsofs) = -tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dx3;
			lhs(2,3,aa,lhsofs) = -tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,aa,lhsofs) = -tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,aa,lhsofs) = -tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,aa,lhsofs) = -tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,aa,lhsofs) = -tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,aa,lhsofs) = -tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dx4;
			lhs(3,4,aa,lhsofs) = -tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,aa,lhsofs) = -tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,aa,lhsofs) = -tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,aa,lhsofs) = -tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,aa,lhsofs) = -tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,aa,lhsofs) = -tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dx5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,bb,lhsofs) = 1.0 + tmp1*2.0*njac(0,0,jacofs2) + tmp1*2.0*dx1;
			lhs(0,1,bb,lhsofs) = tmp1*2.0*njac(0,1,jacofs2);
			lhs(0,2,bb,lhsofs) = tmp1*2.0*njac(0,2,jacofs2);
			lhs(0,3,bb,lhsofs) = tmp1*2.0*njac(0,3,jacofs2);
			lhs(0,4,bb,lhsofs) = tmp1*2.0*njac(0,4,jacofs2);

			lhs(1,0,bb,lhsofs) = tmp1*2.0*njac(1,0,jacofs2);
			lhs(1,1,bb,lhsofs) = 1.0 + tmp1*2.0*njac(1,1,jacofs2) + tmp1*2.0*dx2;
			lhs(1,2,bb,lhsofs) = tmp1*2.0*njac(1,2,jacofs2);
			lhs(1,3,bb,lhsofs) = tmp1*2.0*njac(1,3,jacofs2);
			lhs(1,4,bb,lhsofs) = tmp1*2.0*njac(1,4,jacofs2);

			lhs(2,0,bb,lhsofs) = tmp1*2.0*njac(2,0,jacofs2);
			lhs(2,1,bb,lhsofs) = tmp1*2.0*njac(2,1,jacofs2);
			lhs(2,2,bb,lhsofs) = 1.0 + tmp1*2.0*njac(2,2,jacofs2) + tmp1*2.0*dx3;
			lhs(2,3,bb,lhsofs) = tmp1*2.0*njac(2,3,jacofs2);
			lhs(2,4,bb,lhsofs) = tmp1*2.0*njac(2,4,jacofs2);

			lhs(3,0,bb,lhsofs) = tmp1*2.0*njac(3,0,jacofs2);
			lhs(3,1,bb,lhsofs) = tmp1*2.0*njac(3,1,jacofs2);
			lhs(3,2,bb,lhsofs) = tmp1*2.0*njac(3,2,jacofs2);
			lhs(3,3,bb,lhsofs) = 1.0 + tmp1*2.0*njac(3,3,jacofs2) + tmp1*2.0*dx4;
			lhs(3,4,bb,lhsofs) = tmp1*2.0*njac(3,4,jacofs2);

			lhs(4,0,bb,lhsofs) = tmp1*2.0*njac(4,0,jacofs2);
			lhs(4,1,bb,lhsofs) = tmp1*2.0*njac(4,1,jacofs2);
			lhs(4,2,bb,lhsofs) = tmp1*2.0*njac(4,2,jacofs2);
			lhs(4,3,bb,lhsofs) = tmp1*2.0*njac(4,3,jacofs2);
			lhs(4,4,bb,lhsofs) = 1.0 + tmp1*2.0*njac(4,4,jacofs2) + tmp1*2.0*dx5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,cc,lhsofs) = tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dx1;
			lhs(0,1,cc,lhsofs) = tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,cc,lhsofs) = tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,cc,lhsofs) = tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,cc,lhsofs) = tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,cc,lhsofs) = tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,cc,lhsofs) = tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dx2;
			lhs(1,2,cc,lhsofs) = tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,cc,lhsofs) = tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,cc,lhsofs) = tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);

			lhs(2,0,cc,lhsofs) = tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,cc,lhsofs) = tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,cc,lhsofs) = tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dx3;
			lhs(2,3,cc,lhsofs) = tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,cc,lhsofs) = tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,cc,lhsofs) = tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,cc,lhsofs) = tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,cc,lhsofs) = tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,cc,lhsofs) = tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dx4;
			lhs(3,4,cc,lhsofs) = tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,cc,lhsofs) = tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,cc,lhsofs) = tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,cc,lhsofs) = tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,cc,lhsofs) = tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,cc,lhsofs) = tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dx5;
		} 
		lhsofs += SOLVE_BLOCK;
		jacofs = (jacofs + SOLVE_BLOCK) % (2*SOLVE_BLOCK);
	}
}

__global__ static void x_solve_kernel_2 (double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int j, k, m;
	k = blockIdx.x+1;
	j = blockIdx.y+1;
	m = threadIdx.x;
	lhs += (k-1+(j-1)*nz)*5*5*3*nx;
	__shared__ double rtmp[2][5];
	__shared__ double lhsbtmp[5*5], lhsctmp[5*5], lhsatmp[5*5];

	rtmp[0][m] = rhs(m,0,j,k);
	for (int n = 0; n < 5; n++) {
		lhsbtmp[m+5*n] = lhs(m,n,bb,0);
		lhsctmp[m+5*n] = lhs(m,n,cc,0);
	}
	__syncthreads();
	binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[0]);
	for (int n = 0; n < 5; n++) lhs(m,n,cc,0) = lhsctmp[m+5*n];
	for (int i = 1; i < nx-1; i++) {
		rtmp[1][m] = rhs(m,i,j,k);
		for (int n = 0; n < 5; n++) {
			lhsatmp[m+5*n] = lhs(m,n,aa,i);
			lhsbtmp[m+5*n] = lhs(m,n,bb,i);
		}
		__syncthreads();
		matvec_sub_kernel (m, lhsatmp, rtmp[0], rtmp[1]);
		matmul_sub_kernel (m, lhsatmp, lhsctmp, lhsbtmp);
		for (int n = 0; n < 5; n++) lhsctmp[m+5*n] = lhs(m,n,cc,i);
		__syncthreads();
		binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[1]);
		for (int n = 0; n < 5; n++) lhs(m,n,cc,i) = lhsctmp[m+5*n];
		rhs(m,i-1,j,k) = rtmp[0][m];
		rtmp[0][m] = rtmp[1][m];
	}
	rtmp[1][m] = rhs(m,nx-1,j,k);
	for (int n = 0; n < 5; n++) {
		lhsatmp[m+5*n] = lhs(m,n,aa,nx-1);
		lhsbtmp[m+5*n] = lhs(m,n,bb,nx-1);
	}
	__syncthreads();
	matvec_sub_kernel (m, lhsatmp, rtmp[0], rtmp[1]);
	matvec_sub_kernel (m, lhsatmp, lhsctmp, lhsbtmp);
	binvrhs_kernel (m, lhsbtmp, rtmp[1]);
	for (int i = nx-2; i >= 0; i--) {
		for (int n = 0; n < 5; n++) rtmp[0][m] -= lhs(m,n,cc,i)*rtmp[1][n];
		rhs(m,i,j,k) = rtmp[1][m] = rtmp[0][m];
		if (i > 0) rtmp[0][m] = rhs(m,i-1,j,k);
		__syncthreads();
	}
}
#undef lhs

void BT::x_solve () {
	dim3 grid(nz-2,ny-2);

	START_TIMER(t_xsolve);
	x_solve_kernel_1<<<grid,SOLVE_BLOCK>>>(rho_i, qs, square, u, rhs, lhs, nx, ny, nz); 
	x_solve_kernel_2<<<grid,5>>>(rhs, lhs, nx, ny, nz); 
	STOP_TIMER(t_xsolve);
}

//---------------------------------------------------------------------
//     Performs line solves in Y direction by first factoring
//     the block-tridiagonal matrix into an upper triangular matrix, 
//     and then performing back substitution to solve for the unknow
//     vectors of each line.  
//     
//     Make sure we treat elements zero to cell_size in the direction
//     of the sweep.
//---------------------------------------------------------------------
#define lhs(a,b,c,i) lhs[(a)+5*((i)+ny*((b)+5*(c)))]
__global__ static void y_solve_kernel_1 (const double *rho_i, const double *qs, const double *square, const double *u, double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int i, k, jacofs, jacofs2, lhsofs;
	double tmp1, tmp2, tmp3, utmp[5];
	k = blockIdx.x+1;
	i = blockIdx.y+1;
	__shared__ double fjac[2*5*5*SOLVE_BLOCK];
	__shared__ double njac[2*5*5*SOLVE_BLOCK];
	lhs += (k-1+(i-1)*nz)*5*5*3*ny;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//     This function computes the left hand side for the three y-factors   
	//     jsize = ny-1
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     Compute the indices for storing the tri-diagonal matrix;
	//     determine a (labeled f) and n jacobians for cell c
	//---------------------------------------------------------------------
	jacofs = threadIdx.x;
	lhsofs = jacofs-1;
	while (lhsofs < ny-1) {
		jacofs2 = lhsofs+1;
		tmp1 = rho_i(i,jacofs2,k);
		tmp2 = tmp1 * tmp1;
		tmp3 = tmp1 * tmp2;
		utmp[0] = u(0,i,jacofs2,k);
		utmp[1] = u(1,i,jacofs2,k);
		utmp[2] = u(2,i,jacofs2,k);
		utmp[3] = u(3,i,jacofs2,k);
		utmp[4] = u(4,i,jacofs2,k);

		fjac(0,0,jacofs) = 0.0;
		fjac(0,1,jacofs) = 0.0;
		fjac(0,2,jacofs) = 1.0;
		fjac(0,3,jacofs) = 0.0;
		fjac(0,4,jacofs) = 0.0;

		fjac(1,0,jacofs) = -(utmp[1]*utmp[2])*tmp2;
		fjac(1,1,jacofs) = utmp[2]* tmp1;
		fjac(1,2,jacofs) = utmp[1]* tmp1;
		fjac(1,3,jacofs) = 0.0;
		fjac(1,4,jacofs) = 0.0;

		fjac(2,0,jacofs) = -(utmp[2]*utmp[2]*tmp2) + c2*qs(i,jacofs2,k);
		fjac(2,1,jacofs) = -c2*utmp[1]*tmp1;
		fjac(2,2,jacofs) = (2.0-c2) * utmp[2] * tmp1;
		fjac(2,3,jacofs) = -c2 * utmp[3] * tmp1;
		fjac(2,4,jacofs) = c2;

		fjac(3,0,jacofs) = -(utmp[2]*utmp[3]) * tmp2;
		fjac(3,1,jacofs) = 0.0;
		fjac(3,2,jacofs) = utmp[3] * tmp1;
		fjac(3,3,jacofs) = utmp[2] * tmp1;
		fjac(3,4,jacofs) = 0.0;

		fjac(4,0,jacofs) = (c2*2.0*square(i,jacofs2,k) - c1*utmp[4]) * utmp[2] * tmp2;
		fjac(4,1,jacofs) = -c2 * utmp[1] * utmp[2] * tmp2;
		fjac(4,2,jacofs) = c1 * utmp[4] * tmp1 - c2 * (qs(i,jacofs2,k)+utmp[2]*utmp[2]*tmp2);
		fjac(4,3,jacofs) = -c2 * (utmp[2]*utmp[3]) * tmp2;
		fjac(4,4,jacofs) = c1 * utmp[2] * tmp1;

		njac(0,0,jacofs) = 0.0;
		njac(0,1,jacofs) = 0.0;
		njac(0,2,jacofs) = 0.0;
		njac(0,3,jacofs) = 0.0;
		njac(0,4,jacofs) = 0.0;

		njac(1,0,jacofs) = -c3c4 * tmp2 * utmp[1];
		njac(1,1,jacofs) = c3c4 * tmp1;
		njac(1,2,jacofs) = 0.0;
		njac(1,3,jacofs) = 0.0;
		njac(1,4,jacofs) = 0.0;

		njac(2,0,jacofs) = -con43 * c3c4 * tmp2 * utmp[2];
		njac(2,1,jacofs) = 0.0;
		njac(2,2,jacofs) = con43 * c3c4 * tmp1;
		njac(2,3,jacofs) = 0.0;
		njac(2,4,jacofs) = 0.0;

		njac(3,0,jacofs) = -c3c4 * tmp2 * utmp[3];
		njac(3,1,jacofs) = 0.0;
		njac(3,2,jacofs) = 0.0;
		njac(3,3,jacofs) = c3c4 * tmp1;
		njac(3,4,jacofs) = 0.0;

		njac(4,0,jacofs) = -(c3c4-c1345)*tmp3*(utmp[1]*utmp[1]) - (con43*c3c4-c1345)*tmp3*(utmp[2]*utmp[2]) - (c3c4-c1345)*tmp3*(utmp[3]*utmp[3]) - c1345*tmp2*utmp[4];
		njac(4,1,jacofs) = (c3c4-c1345)*tmp2*utmp[1];
		njac(4,2,jacofs) = (con43*c3c4-c1345) * tmp2 * utmp[2];
		njac(4,3,jacofs) = (c3c4-c1345) * tmp2 * utmp[3];
		njac(4,4,jacofs) = (c1345) * tmp1;
		//---------------------------------------------------------------------
		//     now joacobians set, so form left hand side in y direction
		//---------------------------------------------------------------------
		__syncthreads();
		if (lhsofs < 1) {
			jacofs2 = lhsofs == 0 ? 0 : ny-1;
			for (int m = 0; m < 5; m++) {
				for (int n = 0; n < 5; n++) {
					lhs(m,n,aa,jacofs2) = 0.0;
					lhs(m,n,bb,jacofs2) = 0.0;
					lhs(m,n,cc,jacofs2) = 0.0;
				}
				lhs(m,m,bb,jacofs2) = 1.0;
			}
		} else {
			tmp1 = dt * ty1;
			tmp2 = dt * ty2;

			jacofs2 = (2*SOLVE_BLOCK+jacofs-2) % (2*SOLVE_BLOCK);
			lhs(0,0,aa,lhsofs) = -tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dy1;
			lhs(0,1,aa,lhsofs) = -tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,aa,lhsofs) = -tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,aa,lhsofs) = -tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,aa,lhsofs) = -tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,aa,lhsofs) = -tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,aa,lhsofs) = -tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dy2;
			lhs(1,2,aa,lhsofs) = -tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,aa,lhsofs) = -tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,aa,lhsofs) = -tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);

			lhs(2,0,aa,lhsofs) = -tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,aa,lhsofs) = -tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,aa,lhsofs) = -tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dy3;
			lhs(2,3,aa,lhsofs) = -tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,aa,lhsofs) = -tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,aa,lhsofs) = -tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,aa,lhsofs) = -tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,aa,lhsofs) = -tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,aa,lhsofs) = -tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dy4;
			lhs(3,4,aa,lhsofs) = -tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,aa,lhsofs) = -tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,aa,lhsofs) = -tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,aa,lhsofs) = -tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,aa,lhsofs) = -tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,aa,lhsofs) = -tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dy5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,bb,lhsofs) = 1.0 + tmp1*2.0*njac(0,0,jacofs2) + tmp1*2.0*dy1;
			lhs(0,1,bb,lhsofs) = tmp1*2.0*njac(0,1,jacofs2);
			lhs(0,2,bb,lhsofs) = tmp1*2.0*njac(0,2,jacofs2);
			lhs(0,3,bb,lhsofs) = tmp1*2.0*njac(0,3,jacofs2);
			lhs(0,4,bb,lhsofs) = tmp1*2.0*njac(0,4,jacofs2);

			lhs(1,0,bb,lhsofs) = tmp1*2.0*njac(1,0,jacofs2);
			lhs(1,1,bb,lhsofs) = 1.0 + tmp1*2.0*njac(1,1,jacofs2) + tmp1*2.0*dy2;
			lhs(1,2,bb,lhsofs) = tmp1*2.0*njac(1,2,jacofs2);
			lhs(1,3,bb,lhsofs) = tmp1*2.0*njac(1,3,jacofs2);
			lhs(1,4,bb,lhsofs) = tmp1*2.0*njac(1,4,jacofs2);

			lhs(2,0,bb,lhsofs) = tmp1*2.0*njac(2,0,jacofs2);
			lhs(2,1,bb,lhsofs) = tmp1*2.0*njac(2,1,jacofs2);
			lhs(2,2,bb,lhsofs) = 1.0 + tmp1*2.0*njac(2,2,jacofs2) + tmp1*2.0*dy3;
			lhs(2,3,bb,lhsofs) = tmp1*2.0*njac(2,3,jacofs2);
			lhs(2,4,bb,lhsofs) = tmp1*2.0*njac(2,4,jacofs2);

			lhs(3,0,bb,lhsofs) = tmp1*2.0*njac(3,0,jacofs2);
			lhs(3,1,bb,lhsofs) = tmp1*2.0*njac(3,1,jacofs2);
			lhs(3,2,bb,lhsofs) = tmp1*2.0*njac(3,2,jacofs2);
			lhs(3,3,bb,lhsofs) = 1.0 + tmp1*2.0*njac(3,3,jacofs2) + tmp1*2.0*dy4;
			lhs(3,4,bb,lhsofs) = tmp1*2.0*njac(3,4,jacofs2);

			lhs(4,0,bb,lhsofs) = tmp1*2.0*njac(4,0,jacofs2);
			lhs(4,1,bb,lhsofs) = tmp1*2.0*njac(4,1,jacofs2);
			lhs(4,2,bb,lhsofs) = tmp1*2.0*njac(4,2,jacofs2);
			lhs(4,3,bb,lhsofs) = tmp1*2.0*njac(4,3,jacofs2);
			lhs(4,4,bb,lhsofs) = 1.0 + tmp1*2.0*njac(4,4,jacofs2) + tmp1*2.0*dy5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,cc,lhsofs) = tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dy1;
			lhs(0,1,cc,lhsofs) = tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,cc,lhsofs) = tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,cc,lhsofs) = tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,cc,lhsofs) = tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,cc,lhsofs) = tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,cc,lhsofs) = tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dy2;
			lhs(1,2,cc,lhsofs) = tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,cc,lhsofs) = tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,cc,lhsofs) = tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);

			lhs(2,0,cc,lhsofs) = tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,cc,lhsofs) = tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,cc,lhsofs) = tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dy3;
			lhs(2,3,cc,lhsofs) = tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,cc,lhsofs) = tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,cc,lhsofs) = tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,cc,lhsofs) = tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,cc,lhsofs) = tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,cc,lhsofs) = tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dy4;
			lhs(3,4,cc,lhsofs) = tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,cc,lhsofs) = tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,cc,lhsofs) = tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,cc,lhsofs) = tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,cc,lhsofs) = tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,cc,lhsofs) = tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dy5;
		}
		lhsofs += SOLVE_BLOCK;
		jacofs = (jacofs + SOLVE_BLOCK) % (2*SOLVE_BLOCK);
	}
}

__global__ static void y_solve_kernel_2 (double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int i, k, m;
	k = blockIdx.x+1;
	i = blockIdx.y+1;
	m = threadIdx.x;
	lhs += (k-1+(i-1)*nz)*5*5*3*ny;
	__shared__ double rtmp[2][5];
	__shared__ double lhsbtmp[5*5], lhsctmp[5*5], lhsatmp[5*5];

	rtmp[0][m] = rhs(m,i,0,k);
	for (int n = 0; n < 5; n++) {
		lhsbtmp[m+5*n] = lhs(m,n,bb,0);
		lhsctmp[m+5*n] = lhs(m,n,cc,0);
	}
	__syncthreads();
	binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[0]);
	for (int n = 0; n < 5; n++) lhs(m,n,cc,0) = lhsctmp[m+5*n];
	for (int j = 1; j < ny-1; j++) {
		rtmp[1][m] = rhs(m,i,j,k);
		for (int n = 0; n < 5; n++) {
			lhsatmp[m+5*n] = lhs(m,n,aa,j);
			lhsbtmp[m+5*n] = lhs(m,n,bb,j);
		}
		__syncthreads();
		matvec_sub_kernel(m, lhsatmp, rtmp[0], rtmp[1]);
		matmul_sub_kernel(m, lhsatmp, lhsctmp, lhsbtmp);
		for (int n = 0; n < 5; n++) lhsctmp[m+5*n] = lhs(m,n,cc,j);
		__syncthreads();
		binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[1]);
		for (int n = 0; n < 5; n++) lhs(m,n,cc,j) = lhsctmp[m+5*n];
		rhs(m,i,j-1,k) = rtmp[0][m];
		rtmp[0][m] = rtmp[1][m];
	}
	rtmp[1][m] = rhs(m,i,ny-1,k);
	for (int n = 0; n < 5; n++) {
		lhsatmp[m+5*n] = lhs(m,n,aa,ny-1);
		lhsbtmp[m+5*n] = lhs(m,n,bb,ny-1);
	}
	__syncthreads();
	matvec_sub_kernel(m, lhsatmp, rtmp[0], rtmp[1]);
	matmul_sub_kernel(m, lhsatmp, lhsctmp, lhsbtmp);
	binvrhs_kernel(m, lhsbtmp, rtmp[1]);
	rhs(m,i,ny-1,k) = rtmp[1][m];
	for (int j = ny-2; j >= 0; j--) {
		for (int n = 0; n < 5; n++) rtmp[0][m] -= lhs(m,n,cc,j)*rtmp[1][n];
		rhs(m,i,j,k) = rtmp[1][m] = rtmp[0][m];
		if (j > 0) rtmp[0][m] = rhs(m,i,j-1,k);
		__syncthreads();
	}
}
#undef lhs

void BT::y_solve () {
	dim3 grid2(nz-2,nx-2);

	START_TIMER(t_ysolve);
	y_solve_kernel_1<<<grid2,SOLVE_BLOCK>>>(rho_i, qs, square, u, rhs, lhs, nx, ny, nz); 
	y_solve_kernel_2<<<grid2,5>>>(rhs, lhs, nx, ny, nz);
	STOP_TIMER(t_ysolve);
}

//---------------------------------------------------------------------
//     Performs line solves in Z direction by first factoring the block-tridiagonal matrix into an upper triangular matrix, 
//     and then performing back substitution to solve for the unknow vectors of each line.  
//     
//     Make sure we treat elements zero to cell_size in the direction of the sweep.
//---------------------------------------------------------------------
#define lhs(a,b,c,i) lhs[(a)+5*((i)+nz*((b)+5*(c)))]
__global__ static void z_solve_kernel_1 (const double *qs, const double *square, const double *u, double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int i, j, jacofs, jacofs2, lhsofs;
	double tmp1, tmp2, tmp3, utmp[5];
	i = blockIdx.x+1;
	j = blockIdx.y+1;
	__shared__ double fjac[2*5*5*SOLVE_BLOCK];
	__shared__ double njac[2*5*5*SOLVE_BLOCK];
	lhs += (i-1+(j-1)*nx)*5*5*3*nz;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//     This function computes the left hand side for the three z-factors   
	//     ksize = nz-1
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     Compute the indices for storing the block-diagonal matrix; determine c (labeled f) and s jacobians
	//---------------------------------------------------------------------
	lhsofs = threadIdx.x-1;
	jacofs = threadIdx.x;
	while (lhsofs < nz-1) {
		jacofs2 = lhsofs+1;
		utmp[0] = u(0,i,j,jacofs2);
		utmp[1] = u(1,i,j,jacofs2);
		utmp[2] = u(2,i,j,jacofs2);
		utmp[3] = u(3,i,j,jacofs2);
		utmp[4] = u(4,i,j,jacofs2);
		tmp1 = 1.0/utmp[0];
		tmp2 = tmp1*tmp1;
		tmp3 = tmp1*tmp2;

		fjac(0,0,jacofs) = 0.0;
		fjac(0,1,jacofs) = 0.0;
		fjac(0,2,jacofs) = 0.0;
		fjac(0,3,jacofs) = 1.0;
		fjac(0,4,jacofs) = 0.0;

		fjac(1,0,jacofs) = -(utmp[1]*utmp[3]) * tmp2;
		fjac(1,1,jacofs) = utmp[3] * tmp1;
		fjac(1,2,jacofs) = 0.0;
		fjac(1,3,jacofs) = utmp[1] * tmp1;
		fjac(1,4,jacofs) = 0.0;

		fjac(2,0,jacofs) = -(utmp[2]*utmp[3]) * tmp2;
		fjac(2,1,jacofs) = 0.0;
		fjac(2,2,jacofs) = utmp[3] * tmp1;
		fjac(2,3,jacofs) = utmp[2] * tmp1;
		fjac(2,4,jacofs) = 0.0;

		fjac(3,0,jacofs) = -(utmp[3]*utmp[3]*tmp2) + c2*qs(i,j,jacofs2);
		fjac(3,1,jacofs) = -c2*utmp[1]*tmp1;
		fjac(3,2,jacofs) = -c2*utmp[2]*tmp1;
		fjac(3,3,jacofs) = (2.0-c2) * utmp[3] * tmp1;
		fjac(3,4,jacofs) = c2;

		fjac(4,0,jacofs) = (c2*2.0*square(i,j,jacofs2) - c1*utmp[4]) * utmp[3] * tmp2;
		fjac(4,1,jacofs) = -c2 * (utmp[1]*utmp[3]) * tmp2;
		fjac(4,2,jacofs) = -c2 * (utmp[2]*utmp[3]) * tmp2;
		fjac(4,3,jacofs) = c1 * (utmp[4]*tmp1) - c2 * (qs(i,j,jacofs2)+utmp[3]*utmp[3]*tmp2);
		fjac(4,4,jacofs) = c1 * utmp[3] * tmp1;

		njac(0,0,jacofs) = 0.0;
		njac(0,1,jacofs) = 0.0;
		njac(0,2,jacofs) = 0.0;
		njac(0,3,jacofs) = 0.0;
		njac(0,4,jacofs) = 0.0;

		njac(1,0,jacofs) = -c3c4 * tmp2 * utmp[1];
		njac(1,1,jacofs) = c3c4 * tmp1;
		njac(1,2,jacofs) = 0.0;
		njac(1,3,jacofs) = 0.0;
		njac(1,4,jacofs) = 0.0;

		njac(2,0,jacofs) = -c3c4 * tmp2 * utmp[2];
		njac(2,1,jacofs) = 0.0;
		njac(2,2,jacofs) = c3c4 * tmp1;
		njac(2,3,jacofs) = 0.0;
		njac(2,4,jacofs) = 0.0;

		njac(3,0,jacofs) = -con43 * c3c4 * tmp2 * utmp[3];
		njac(3,1,jacofs) = 0.0;
		njac(3,2,jacofs) = 0.0;
		njac(3,3,jacofs) = con43 * c3 * c4 * tmp1;
		njac(3,4,jacofs) = 0.0;

		njac(4,0,jacofs) = -(c3c4-c1345)*tmp3*(utmp[1]*utmp[1]) - (c3c4-c1345)*tmp3*(utmp[2]*utmp[2]) - (con43*c3c4-c1345)*tmp3*(utmp[3]*utmp[3]) - c1345*tmp2*utmp[4];
		njac(4,1,jacofs) = (c3c4-c1345)*tmp2*utmp[1];
		njac(4,2,jacofs) = (c3c4-c1345)*tmp2*utmp[2];
		njac(4,3,jacofs) = (con43*c3c4-c1345)*tmp2*utmp[3];
		njac(4,4,jacofs) = c1345 * tmp1;
		//---------------------------------------------------------------------
		//     now jacobians set, so form left hand side in z direction
		//---------------------------------------------------------------------
		__syncthreads();
		if (lhsofs < 1) {
			jacofs2 = lhsofs == 0 ? 0 : nz-1;
			for (int m = 0; m < 5; m++) {
				for (int n = 0; n < 5; n++) {
					lhs(m,n,aa,jacofs2) = 0.0;
					lhs(m,n,bb,jacofs2) = 0.0;
					lhs(m,n,cc,jacofs2) = 0.0;
				}
				lhs(m,m,bb,jacofs2) = 1.0;
			}
		} else {
			tmp1 = dt*tz1;
			tmp2 = dt*tz2;

			jacofs2 = (2*SOLVE_BLOCK+jacofs-2) % (2*SOLVE_BLOCK);
			lhs(0,0,aa,lhsofs) = -tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dz1;
			lhs(0,1,aa,lhsofs) = -tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,aa,lhsofs) = -tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,aa,lhsofs) = -tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,aa,lhsofs) = -tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,aa,lhsofs) = -tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,aa,lhsofs) = -tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dz2;
			lhs(1,2,aa,lhsofs) = -tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,aa,lhsofs) = -tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,aa,lhsofs) = -tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);
				
			lhs(2,0,aa,lhsofs) = -tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,aa,lhsofs) = -tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,aa,lhsofs) = -tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dz3;
			lhs(2,3,aa,lhsofs) = -tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,aa,lhsofs) = -tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,aa,lhsofs) = -tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,aa,lhsofs) = -tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,aa,lhsofs) = -tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,aa,lhsofs) = -tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dz4;
			lhs(3,4,aa,lhsofs) = -tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,aa,lhsofs) = -tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,aa,lhsofs) = -tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,aa,lhsofs) = -tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,aa,lhsofs) = -tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,aa,lhsofs) = -tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dz5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,bb,lhsofs) = 1.0 + tmp1*2.0*njac(0,0,jacofs2) + tmp1*2.0*dz1;
			lhs(0,1,bb,lhsofs) = tmp1*2.0*njac(0,1,jacofs2);
			lhs(0,2,bb,lhsofs) = tmp1*2.0*njac(0,2,jacofs2);
			lhs(0,3,bb,lhsofs) = tmp1*2.0*njac(0,3,jacofs2);
			lhs(0,4,bb,lhsofs) = tmp1*2.0*njac(0,4,jacofs2);

			lhs(1,0,bb,lhsofs) = tmp1*2.0*njac(1,0,jacofs2);
			lhs(1,1,bb,lhsofs) = 1.0 + tmp1*2.0*njac(1,1,jacofs2) + tmp1*2.0*dz2;
			lhs(1,2,bb,lhsofs) = tmp1*2.0*njac(1,2,jacofs2);
			lhs(1,3,bb,lhsofs) = tmp1*2.0*njac(1,3,jacofs2);
			lhs(1,4,bb,lhsofs) = tmp1*2.0*njac(1,4,jacofs2);

			lhs(2,0,bb,lhsofs) = tmp1*2.0*njac(2,0,jacofs2);
			lhs(2,1,bb,lhsofs) = tmp1*2.0*njac(2,1,jacofs2);
			lhs(2,2,bb,lhsofs) = 1.0 + tmp1*2.0*njac(2,2,jacofs2) + tmp1*2.0*dz3;
			lhs(2,3,bb,lhsofs) = tmp1*2.0*njac(2,3,jacofs2);
			lhs(2,4,bb,lhsofs) = tmp1*2.0*njac(2,4,jacofs2);

			lhs(3,0,bb,lhsofs) = tmp1*2.0*njac(3,0,jacofs2);
			lhs(3,1,bb,lhsofs) = tmp1*2.0*njac(3,1,jacofs2);
			lhs(3,2,bb,lhsofs) = tmp1*2.0*njac(3,2,jacofs2);
			lhs(3,3,bb,lhsofs) = 1.0 + tmp1*2.0*njac(3,3,jacofs2) + tmp1*2.0*dz4;
			lhs(3,4,bb,lhsofs) = tmp1*2.0*njac(3,4,jacofs2);

			lhs(4,0,bb,lhsofs) = tmp1*2.0*njac(4,0,jacofs2);
			lhs(4,1,bb,lhsofs) = tmp1*2.0*njac(4,1,jacofs2);
			lhs(4,2,bb,lhsofs) = tmp1*2.0*njac(4,2,jacofs2);
			lhs(4,3,bb,lhsofs) = tmp1*2.0*njac(4,3,jacofs2);
			lhs(4,4,bb,lhsofs) = 1.0 + tmp1*2.0*njac(4,4,jacofs2) + tmp1*2.0*dz5;

			jacofs2 = (jacofs2+1) % (2*SOLVE_BLOCK);
			lhs(0,0,cc,lhsofs) = tmp2*fjac(0,0,jacofs2) - tmp1*njac(0,0,jacofs2) - tmp1*dz1;
			lhs(0,1,cc,lhsofs) = tmp2*fjac(0,1,jacofs2) - tmp1*njac(0,1,jacofs2);
			lhs(0,2,cc,lhsofs) = tmp2*fjac(0,2,jacofs2) - tmp1*njac(0,2,jacofs2);
			lhs(0,3,cc,lhsofs) = tmp2*fjac(0,3,jacofs2) - tmp1*njac(0,3,jacofs2);
			lhs(0,4,cc,lhsofs) = tmp2*fjac(0,4,jacofs2) - tmp1*njac(0,4,jacofs2);

			lhs(1,0,cc,lhsofs) = tmp2*fjac(1,0,jacofs2) - tmp1*njac(1,0,jacofs2);
			lhs(1,1,cc,lhsofs) = tmp2*fjac(1,1,jacofs2) - tmp1*njac(1,1,jacofs2) - tmp1*dz2;
			lhs(1,2,cc,lhsofs) = tmp2*fjac(1,2,jacofs2) - tmp1*njac(1,2,jacofs2);
			lhs(1,3,cc,lhsofs) = tmp2*fjac(1,3,jacofs2) - tmp1*njac(1,3,jacofs2);
			lhs(1,4,cc,lhsofs) = tmp2*fjac(1,4,jacofs2) - tmp1*njac(1,4,jacofs2);

			lhs(2,0,cc,lhsofs) = tmp2*fjac(2,0,jacofs2) - tmp1*njac(2,0,jacofs2);
			lhs(2,1,cc,lhsofs) = tmp2*fjac(2,1,jacofs2) - tmp1*njac(2,1,jacofs2);
			lhs(2,2,cc,lhsofs) = tmp2*fjac(2,2,jacofs2) - tmp1*njac(2,2,jacofs2) - tmp1*dz3;
			lhs(2,3,cc,lhsofs) = tmp2*fjac(2,3,jacofs2) - tmp1*njac(2,3,jacofs2);
			lhs(2,4,cc,lhsofs) = tmp2*fjac(2,4,jacofs2) - tmp1*njac(2,4,jacofs2);

			lhs(3,0,cc,lhsofs) = tmp2*fjac(3,0,jacofs2) - tmp1*njac(3,0,jacofs2);
			lhs(3,1,cc,lhsofs) = tmp2*fjac(3,1,jacofs2) - tmp1*njac(3,1,jacofs2);
			lhs(3,2,cc,lhsofs) = tmp2*fjac(3,2,jacofs2) - tmp1*njac(3,2,jacofs2);
			lhs(3,3,cc,lhsofs) = tmp2*fjac(3,3,jacofs2) - tmp1*njac(3,3,jacofs2) - tmp1*dz4;
			lhs(3,4,cc,lhsofs) = tmp2*fjac(3,4,jacofs2) - tmp1*njac(3,4,jacofs2);

			lhs(4,0,cc,lhsofs) = tmp2*fjac(4,0,jacofs2) - tmp1*njac(4,0,jacofs2);
			lhs(4,1,cc,lhsofs) = tmp2*fjac(4,1,jacofs2) - tmp1*njac(4,1,jacofs2);
			lhs(4,2,cc,lhsofs) = tmp2*fjac(4,2,jacofs2) - tmp1*njac(4,2,jacofs2);
			lhs(4,3,cc,lhsofs) = tmp2*fjac(4,3,jacofs2) - tmp1*njac(4,3,jacofs2);
			lhs(4,4,cc,lhsofs) = tmp2*fjac(4,4,jacofs2) - tmp1*njac(4,4,jacofs2) - tmp1*dz5;
		}
		lhsofs += SOLVE_BLOCK;
		jacofs = (jacofs + SOLVE_BLOCK) % (2*SOLVE_BLOCK);
	}
}

__global__ static void z_solve_kernel_2 (double *rhs, double *lhs, const int nx, const int ny, const int nz) {
	int i, j, m;
	i = blockIdx.x+1;
	j = blockIdx.y+1;
	m = threadIdx.x;
	lhs += (i-1+(j-1)*nx)*5*5*3*nz;
	__shared__ double rtmp[2][5];
	__shared__ double lhsbtmp[5*5], lhsctmp[5*5], lhsatmp[5*5];

	// copy date to shared memory
	rtmp[0][m] = rhs(m,i,j,0);
	for (int n = 0; n < 5; n++) {
		lhsbtmp[m+5*n] = lhs(m,n,bb,0);
		lhsctmp[m+5*n] = lhs(m,n,cc,0);
	}
	__syncthreads();
	binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[0]);
	for (int n = 0; n < 5; n++) lhs(m,n,cc,0) = lhsctmp[m+5*n];
	for (int k = 1; k < nz-1; k++) {
		rtmp[1][m] = rhs(m,i,j,k);
		for (int n = 0; n < 5; n++) {
			lhsatmp[m+5*n] = lhs(m,n,aa,k);
			lhsbtmp[m+5*n] = lhs(m,n,bb,k);
		}
		__syncthreads();
		matvec_sub_kernel (m, lhsatmp, rtmp[0], rtmp[1]);
		matmul_sub_kernel (m, lhsatmp, lhsctmp, lhsbtmp);
		for (int n = 0; n < 5; n++) lhsctmp[m+5*n] = lhs(m,n,cc,k);
		__syncthreads();
		binvcrhs_kernel (m, lhsbtmp, lhsctmp, rtmp[1]);
		for (int n = 0; n < 5; n++) lhs(m,n,cc,k) = lhsctmp[m+5*n];
		rhs(m,i,j,k-1) = rtmp[0][m];
		rtmp[0][m] = rtmp[1][m];
	}
	rtmp[1][m] = rhs(m,i,j,nz-1);
	for (int n = 0; n < 5; n++) {
		lhsatmp[m+5*n] = lhs(m,n,aa,nz-1);
		lhsbtmp[m+5*n] = lhs(m,n,bb,nz-1);
	}
	__syncthreads();
	matvec_sub_kernel (m, lhsatmp, rtmp[0], rtmp[1]);
	matmul_sub_kernel (m, lhsatmp, lhsctmp, lhsbtmp);
	binvrhs_kernel (m, lhsbtmp, rtmp[1]);
	rhs(m,i,j,nz-1) = rtmp[1][m];
	for (int k = nz-2; k >= 0; k--) {
		for (int n = 0; n < 5; n++) rtmp[0][m] -= lhs(m,n,cc,k)*rtmp[1][n];
		rhs(m,i,j,k) = rtmp[1][m] = rtmp[0][m];
		if (k > 0) rtmp[0][m] = rhs(m,i,j,k-1);
		__syncthreads();
	}
}
#undef lhs

void BT::z_solve () {
	dim3 grid2(nx-2,ny-2);

	START_TIMER(t_zsolve);
	z_solve_kernel_1<<<grid2,SOLVE_BLOCK>>>(qs, square, u, rhs, lhs, nx, ny, nz);  
	z_solve_kernel_2<<<grid2,5>>>(rhs, lhs, nx, ny, nz);
	STOP_TIMER(t_zsolve);
}

//---------------------------------------------------------------------
//     this function returns the exact solution at point xi, eta, zeta  
//---------------------------------------------------------------------
__device__ static void exact_solution_kernel (const double xi, const double eta, const double zeta, double *dtemp) {
	using namespace gpu_mod;

	for (int m = 0; m < 5; m++) 
		dtemp[m] = ce[0][m] + xi*(ce[1][m] + xi*(ce[4][m] + xi*(ce[7][m] + xi*ce[10][m]))) + eta*(ce[2][m] + eta*(ce[5][m] + eta*(ce[8][m] + eta*ce[11][m])))+zeta*(ce[3][m] + zeta*(ce[6][m] + zeta*(ce[9][m] + zeta*ce[12][m])));
}

//---------------------------------------------------------------------
//     compute the right hand side based on exact solution
//---------------------------------------------------------------------
__global__ static void exact_rhs_kernel_init (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	for (m = 0; m < 5; m++) forcing(m,i,j,k) = 0.0;
}

__global__ static void exact_rhs_kernel_x (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, dtpp, dtemp[5];
	double ue[5][5], buf[3][5], cuf[3], q[3];

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	j = blockIdx.y*blockDim.y+threadIdx.y+1;

	if (k >= nz-1 || j >= ny-1) return;

	using namespace gpu_mod;

	zeta = (double)k * dnzm1;
	eta = (double)j * dnym1;
	//---------------------------------------------------------------------
	//      xi-direction flux differences                      
	//---------------------------------------------------------------------
	for (i = 0; i < 3; i++) {
		xi = (double)i * dnxm1;
		exact_solution_kernel(xi, eta, zeta, dtemp);
		for (m = 0; m < 5; m++) ue[i+1][m] = dtemp[m];
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[i][m] = dtpp*dtemp[m];
		cuf[i] = buf[i][1] * buf[i][1];
		buf[i][0] = cuf[i] + buf[i][2] * buf[i][2] + buf[i][3] * buf[i][3];
		q[i] = 0.5 * (buf[i][1]*ue[i+1][1] + buf[i][2]*ue[i+1][2] + buf[i][3]*ue[i+1][3]);
	}
	for (i = 1; i < nx-1; i++) {
		if (i+2 < nx) {
			xi = (double)(i+2) * dnxm1;
			exact_solution_kernel(xi, eta, zeta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}

		dtemp[0] = 0.0 - tx2*(ue[3][1]-ue[1][1])+ dx1tx1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = 0.0 - tx2*((ue[3][1]*buf[2][1]+c2*(ue[3][4]-q[2]))-(ue[1][1]*buf[0][1]+c2*(ue[1][4]-q[0])))+xxcon1*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dx2tx1*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2] = 0.0 - tx2*(ue[3][2]*buf[2][1]-ue[1][2]*buf[0][1])+xxcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dx3tx1*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3] = 0.0 - tx2*(ue[3][3]*buf[2][1]-ue[1][3]*buf[0][1])+xxcon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dx4tx1*(ue[3][3]-2.0*ue[2][3]+ue[1][3]);
		dtemp[4] = 0.0 - tx2*(buf[2][1]*(c1*ue[3][4]-c2*q[2])-buf[0][1]*(c1*ue[1][4]-c2*q[0]))+0.5*xxcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+xxcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+
					xxcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dx5tx1*(ue[3][4]-2.0*ue[2][4]+ ue[1][4]);

		//---------------------------------------------------------------------
		//            Fourth-order dissipation                         
		//---------------------------------------------------------------------
		if (i == 1) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(5.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i == 2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i >= 3 && i < nx-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m]+6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (i == nx-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] +6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (i == nx-2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] + 5.0*ue[2][m]);
		}

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (i < nx-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp*ue[3][m];
			cuf[2] = buf[2][1] * buf[2][1];
			buf[2][0] = cuf[2] + buf[2][2] * buf[2][2] + buf[2][3] * buf[2][3];
			q[2] = 0.5 * (buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_kernel_y (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, dtpp, dtemp[5];
	double ue[5][5], buf[3][5], cuf[5], q[5];

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || i >= nx-1) return;
	
	using namespace gpu_mod;

	zeta = (double)k * dnzm1;
	xi = (double)i * dnxm1;
	//---------------------------------------------------------------------
	//  eta-direction flux differences             
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) {
		eta = (double)j * dnym1;
		exact_solution_kernel(xi, eta, zeta, dtemp);
		for (m = 0; m < 5; m++) ue[j+1][m] = dtemp[m];;
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[j][m] = dtpp * dtemp[m];
		cuf[j] = buf[j][2] * buf[j][2];
		buf[j][0] = cuf[j] + buf[j][1] * buf[j][1] + buf[j][3] * buf[j][3];
		q[j] = 0.5*(buf[j][1]*ue[j+1][1] + buf[j][2]*ue[j+1][2] + buf[j][3]*ue[j+1][3]);
	}
	for (j = 1; j < ny-1; j++) {
		if (j+2 < ny) {
			eta = (double)(j+2) * dnym1;
			exact_solution_kernel(xi, eta, zeta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}

		dtemp[0] = forcing(0,i,j,k) - ty2*(ue[3][2]-ue[1][2])+ dy1ty1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = forcing(1,i,j,k) - ty2*(ue[3][1]*buf[2][2]-ue[1][1]*buf[0][2])+yycon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dy2ty1*(ue[3][1]-2.0*ue[2][1]+ ue[1][1]);
		dtemp[2] = forcing(2,i,j,k) - ty2*((ue[3][2]*buf[2][2]+c2*(ue[3][4]-q[2]))-(ue[1][2]*buf[0][2]+c2*(ue[1][4]-q[0])))+yycon1*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dy3ty1*(ue[3][2]-2.0*ue[2][2] +ue[1][2]);
		dtemp[3] = forcing(3,i,j,k) - ty2*(ue[3][3]*buf[2][2]-ue[1][3]*buf[0][2])+yycon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dy4ty1*(ue[3][3]-2.0*ue[2][3]+ ue[1][3]);
		dtemp[4] = forcing(4,i,j,k) - ty2*(buf[2][2]*(c1*ue[3][4]-c2*q[2])-buf[0][2]*(c1*ue[1][4]-c2*q[0]))+0.5*yycon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+yycon4*(cuf[2]-2.0*cuf[1]+cuf[0])+
					yycon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dy5ty1*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		//---------------------------------------------------------------------
		//            Fourth-order dissipation                      
		//---------------------------------------------------------------------
		if (j == 1) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (5.0*ue[2][m] - 4.0*ue[3][m] +ue[4][m]);
		} else if (j == 2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] +       ue[4][m]);
		} else if (j >= 3 && j < ny-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp*(ue[0][m] - 4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m] + ue[4][m]);
		} else if (j == ny-3) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (ue[0][m] - 4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (j == ny-2) {
			for (m = 0; m < 5; m++) forcing(m,i,j,k) = dtemp[m] - dssp * (ue[0][m] - 4.0*ue[1][m] + 5.0*ue[2][m]);
		}

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (j < ny-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp * ue[3][m];
			cuf[2] = buf[2][2] * buf[2][2];
			buf[2][0] = cuf[2] + buf[2][1] * buf[2][1] + buf[2][3] * buf[2][3];
			q[2] = 0.5*(buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

__global__ static void exact_rhs_kernel_z (double *forcing, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, dtpp, dtemp[5];
	double ue[5][5], buf[3][5], cuf[3], q[3];

	j = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (j >= ny-1 || i >= nx-1) return;

	using namespace gpu_mod;

	eta = (double)j * dnym1;
	xi = (double)i * dnxm1;
	//---------------------------------------------------------------------
	//      zeta-direction flux differences                      
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) {
		zeta = (double)k * dnzm1;
		exact_solution_kernel(xi, eta, zeta, dtemp);
		for (m = 0; m < 5; m++) ue[k+1][m] = dtemp[m];
		dtpp = 1.0/dtemp[0];
		for (m = 1; m < 5; m++) buf[k][m] = dtpp * dtemp[m];
		cuf[k] = buf[k][3] * buf[k][3];
		buf[k][0] = cuf[k] + buf[k][1] * buf[k][1] + buf[k][2] * buf[k][2];
		q[k] = 0.5*(buf[k][1]*ue[k+1][1] + buf[k][2]*ue[k+1][2] + buf[k][3]*ue[k+1][3]);
	}

	for (k = 1; k < nz-1; k++) {
		if (k+2 < nz) {
			zeta = (double)(k+2) * dnzm1;
			exact_solution_kernel(xi, eta, zeta, dtemp);
			for (m = 0; m < 5; m++) ue[4][m] = dtemp[m];
		}

		dtemp[0] = forcing(0,i,j,k) - tz2*(ue[3][3]-ue[1][3])+dz1tz1*(ue[3][0]-2.0*ue[2][0]+ue[1][0]);
		dtemp[1] = forcing(1,i,j,k) - tz2*(ue[3][1]*buf[2][3]-ue[1][1]*buf[0][3])+zzcon2*(buf[2][1]-2.0*buf[1][1]+buf[0][1])+dz2tz1*(ue[3][1]-2.0*ue[2][1]+ue[1][1]);
		dtemp[2] = forcing(2,i,j,k) - tz2*(ue[3][2]*buf[2][3]-ue[1][2]*buf[0][3])+zzcon2*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dz3tz1*(ue[3][2]-2.0*ue[2][2]+ue[1][2]);
		dtemp[3] = forcing(3,i,j,k) - tz2*((ue[3][3]*buf[2][3]+c2*(ue[3][4]-q[2]))-(ue[1][3]*buf[0][3]+c2*(ue[1][4]-q[0])))+zzcon1*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dz4tz1*(ue[3][3]-2.0*ue[2][3] +ue[1][3]);
		dtemp[4] = forcing(4,i,j,k) - tz2*(buf[2][3]*(c1*ue[3][4]-c2*q[2])-buf[0][3]*(c1*ue[1][4]-c2*q[0]))+0.5*zzcon3*(buf[2][0]-2.0*buf[1][0]+buf[0][0])+
					zzcon4*(cuf[2]-2.0*cuf[1]+cuf[0])+zzcon5*(buf[2][4]-2.0*buf[1][4]+buf[0][4])+dz5tz1*(ue[3][4]-2.0*ue[2][4]+ue[1][4]);
		//---------------------------------------------------------------------
		//            Fourth-order dissipation
		//---------------------------------------------------------------------
		if (k == 1) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(5.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k == 2) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k >= 3 && k < nz-3) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m]+6.0*ue[2][m]-4.0*ue[3][m]+ue[4][m]);
		} else if (k == nz-3) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m] + 6.0*ue[2][m] - 4.0*ue[3][m]);
		} else if (k == nz-2) {
			for (m = 0; m < 5; m++) dtemp[m] = dtemp[m] - dssp*(ue[0][m]-4.0*ue[1][m]+5.0*ue[2][m]);
		}
		//---------------------------------------------------------------------
		// now change the sign of the forcing function, 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) forcing(m,i,j,k) = -1.0 * dtemp[m];

		for (m = 0; m < 5; m++) {
			ue[0][m] = ue[1][m]; 
			ue[1][m] = ue[2][m];
			ue[2][m] = ue[3][m];
			ue[3][m] = ue[4][m];
			buf[0][m] = buf[1][m];
			buf[1][m] = buf[2][m];
		}
		cuf[0] = cuf[1]; cuf[1] = cuf[2];
		q[0] = q[1]; q[1] = q[2];

		if (k < nz-2) {
			dtpp = 1.0/ue[3][0];
			for (m = 1; m < 5; m++) buf[2][m] = dtpp * ue[3][m];
			cuf[2] = buf[2][3] * buf[2][3];
			buf[2][0] = cuf[2] + buf[2][1] * buf[2][1] + buf[2][2] * buf[2][2];
			q[2] = 0.5*(buf[2][1]*ue[3][1] + buf[2][2]*ue[3][2] + buf[2][3]*ue[3][3]);
		}
	}
}

void BT::exact_rhs () {
	dim3 gridyz(ny,nz);
	exact_rhs_kernel_init<<<gridyz,nx>>>(forcing, nx, ny, nz);

	int yblock = min(ERHS_BLOCK,ny-2);
	int ygrid = (ny-2+yblock-1)/yblock;
	int zblock_y = min(ERHS_BLOCK/yblock,nz-2);
	int zgrid_y = (nz-2+zblock_y-1)/zblock_y;
	dim3 grid_x(zgrid_y,ygrid), block_x(zblock_y,yblock);
	exact_rhs_kernel_x<<<grid_x,block_x>>>(forcing, nx, ny, nz);

	int xblock = min(ERHS_BLOCK,nx-2);
	int xgrid = (nx-2+xblock-1)/xblock;
	int zblock_x = min(ERHS_BLOCK/xblock,nz-2);
	int zgrid_x = (nz-2+zblock_x-1)/zblock_x;
	dim3 grid_y(zgrid_x,xgrid), block_y(zblock_x,xblock);
	exact_rhs_kernel_y<<<grid_y,block_y>>>(forcing, nx, ny, nz);

	int yblock_x = min(ERHS_BLOCK/xblock,ny-2);
	int ygrid_x = (ny-2+yblock_x-1)/yblock_x;
	dim3 grid_z(ygrid_x,xgrid), block_z(yblock_x,xblock);
	exact_rhs_kernel_z<<<grid_z,block_z>>>(forcing, nx, ny, nz);
}

//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
__global__ static void initialize_kernel (double *u, const int nx, const int ny, const int nz) {
	int i, j, k;
	double xi, eta, zeta, temp[5], Pxi, Peta, Pzeta;
	double Pface11[5], Pface12[5], Pface21[5], Pface22[5], Pface31[5], Pface32[5];

	k = blockIdx.x;
	j = blockIdx.y;
	i = threadIdx.x;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//  Later (in compute_rhs) we compute 1/u for every element. A few of 
	//  the corner elements are not used, but it convenient (and faster) 
	//  to compute the whole thing with a simple loop. Make sure those 
	//  values are nonzero by initializing the whole thing here. 
	//---------------------------------------------------------------------
	for (int m = 0; m < 5; m++) u(m,i,j,k) = 1.0;

	//---------------------------------------------------------------------
	// first store the "interpolated" values everywhere on the zone    
	//---------------------------------------------------------------------
	zeta = (double)k * dnzm1;
	eta = (double)j * dnym1;
	xi = (double)i * dnxm1;
	exact_solution_kernel (0.0, eta, zeta, Pface11);
	exact_solution_kernel (1.0, eta, zeta, Pface12);
	exact_solution_kernel (xi, 0.0, zeta, Pface21);
	exact_solution_kernel (xi, 1.0, zeta, Pface22);
	exact_solution_kernel (xi, eta, 0.0, Pface31);
	exact_solution_kernel (xi, eta, 1.0, Pface32);
	for (int m = 0; m < 5; m++) {
		Pxi = xi * Pface12[m] + (1.0-xi)*Pface11[m];
		Peta = eta * Pface22[m] + (1.0-eta)*Pface21[m];
		Pzeta = zeta * Pface32[m] + (1.0-zeta)*Pface31[m];
		u(m,i,j,k) = Pxi + Peta + Pzeta - Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + Pxi*Peta*Pzeta;
	}

	//---------------------------------------------------------------------
	// now store the exact values on the boundaries        
	//---------------------------------------------------------------------

	//---------------------------------------------------------------------
	// west face                                                  
	//---------------------------------------------------------------------
	xi = 0.0;
	if (i == 0) {
		zeta = (double)k * dnzm1;
		eta = (double)j * dnym1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}

	//---------------------------------------------------------------------
	// east face                                                      
	//---------------------------------------------------------------------
	xi = 1.0;
	if (i == nx-1) {
		zeta = (double)k * dnzm1;
		eta = (double)j * dnym1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}

	//---------------------------------------------------------------------
	// south face                                                 
	//---------------------------------------------------------------------
	eta = 0.0;
	if (j == 0) {
		zeta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zeta,temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}

	//---------------------------------------------------------------------
	// north face                                    
	//---------------------------------------------------------------------
	eta = 1.0;
	if (j == ny-1) {
		zeta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zeta,temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}

	//---------------------------------------------------------------------
	// bottom face                                       
	//---------------------------------------------------------------------
	zeta = 0.0;
	if (k == 0) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}

	//---------------------------------------------------------------------
	// top face     
	//---------------------------------------------------------------------
	zeta = 1.0;
	if (k == nz-1) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (int m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
}

void BT::initialize() {
	dim3 grid(nz,ny);

	initialize_kernel<<<grid,nx>>>(u, nx, ny, nz);
}

//---------------------------------------------------------------------
//     this function computes the norm of the difference between the
//     computed solution and the exact solution
//---------------------------------------------------------------------
__global__ static void error_norm_kernel (double *rms, const double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, u_exact[5], rms_loc[5];

	j = blockIdx.x*blockDim.x+threadIdx.x;
	i = blockIdx.y*blockDim.y+threadIdx.y;
	if (j >= ny || i >= nx) return;

	using namespace gpu_mod;

	for (m = 0; m < 5; m++) rms_loc[m] = 0.0;

	xi = (double)i * dnxm1;
	eta = (double)j * dnym1;

	for (k = 0; k < nz; k++) {
		zeta = (double)k * dnzm1;
		exact_solution_kernel (xi, eta, zeta, u_exact);
		for (m = 0; m < 5; m++) {
			double add = u(m,i,j,k) - u_exact[m];
			rms_loc[m] += add*add;
		}
	}

	for (m = 0; m < 5; m++) rms[i+nx*(j+ny*m)] = rms_loc[m];
}

__global__ static void reduce_norm_kernel (double *rms, const int nx, const int ny, const int nz) {
	int i, m, maxpos, dist;
	__shared__ double buffer[NORM_BLOCK][5];

	i = threadIdx.x;
	for (m = 0; m < 5; m++) buffer[i][m] = 0.0;

	while (i < nx*ny) {
		for (m = 0; m < 5; m++) buffer[threadIdx.x][m] += rms[i+nx*ny*m];
		i += blockDim.x;
	}

	maxpos = blockDim.x;
	dist = (maxpos+1)/2;
	i = threadIdx.x;
	__syncthreads();
	while (maxpos > 1) {
		if (i < dist && i+dist < maxpos)
			for (m = 0; m < 5; m++) buffer[i][m] += buffer[i+dist][m];
		maxpos = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}
	
	m = threadIdx.x;
	if (m < 5) rms[m] = sqrt(buffer[0][m]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));
}

void BT::error_norm () {
	int xblock = min(64,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int yblock = min(64/xblock,ny);
	int ygrid = (ny+yblock-1)/yblock;
	dim3 grid(ygrid,xgrid), block(yblock,xblock);

	error_norm_kernel<<<grid,block>>>(rmsbuf, u, nx, ny, nz);
	reduce_norm_kernel<<<1,NORM_BLOCK>>>(rmsbuf, nx, ny, nz);
	HANDLE_ERROR(cudaMemcpy(xce, rmsbuf, 5*sizeof(double), cudaMemcpyDeviceToHost));
}

__global__ static void rhs_norm_kernel (double *rms, const double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rms_loc[5];

	j = blockIdx.x*blockDim.x+threadIdx.x;
	i = blockIdx.y*blockDim.y+threadIdx.y;
	if (j >= ny || i >= nx) return;

	for (m = 0; m < 5; m++) rms_loc[m] = 0.0;
	if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1) {
		for (k = 1; k < nz-1; k++) {
			for (int m = 0; m < 5; m++) {
				double add = rhs(m,i,j,k);
				rms_loc[m] += add*add;
			}
		}
	}

	for (m = 0; m < 5; m++) rms[i+nx*(j+ny*m)] = rms_loc[m];

}
void BT::rhs_norm () {
	int xblock = min(64,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int yblock = min(64/xblock,ny);
	int ygrid = (ny+yblock-1)/yblock;
	dim3 grid(ygrid,xgrid), block(yblock,xblock);

	rhs_norm_kernel<<<grid,block>>>(rmsbuf, rhs, nx, ny, nz);
	reduce_norm_kernel<<<1,NORM_BLOCK>>>(rmsbuf, nx, ny, nz);

	HANDLE_ERROR(cudaMemcpy(xcr, rmsbuf, 5*sizeof(double), cudaMemcpyDeviceToHost));
}

void BT::set_constants() {

	double ce[13][5];
	ce[0][0] = 2.0;
	ce[1][0] = 0.0;
	ce[2][0] = 0.0;
	ce[3][0] = 4.0;
	ce[4][0] = 5.0;
	ce[5][0] = 3.0;
	ce[6][0] = 0.5;
	ce[7][0] = 0.02;
	ce[8][0] = 0.01;
	ce[9][0] = 0.03;
	ce[10][0] = 0.5;
	ce[11][0] = 0.4;
	ce[12][0] = 0.3;

	ce[0][1] = 1.0;
	ce[1][1] = 0.0;
	ce[2][1] = 0.0;
	ce[3][1] = 0.0;
	ce[4][1] = 1.0;
	ce[5][1] = 2.0;
	ce[6][1] = 3.0;
	ce[7][1] = 0.01;
	ce[8][1] = 0.03;
	ce[9][1] = 0.02;
	ce[10][1] = 0.4;
	ce[11][1] = 0.3;
	ce[12][1] = 0.5;

	ce[0][2] = 2.0;
	ce[1][2] = 2.0;
	ce[2][2] = 0.0;
	ce[3][2] = 0.0;
	ce[4][2] = 0.0;
	ce[5][2] = 2.0;
	ce[6][2] = 3.0;
	ce[7][2] = 0.04;
	ce[8][2] = 0.03;
	ce[9][2] = 0.05;
	ce[10][2] = 0.3;
	ce[11][2] = 0.5;
	ce[12][2] = 0.4;

	ce[0][3] = 2.0;
	ce[1][3] = 2.0;
	ce[2][3] = 0.0;
	ce[3][3] = 0.0;
	ce[4][3] = 0.0;
	ce[5][3] = 2.0;
	ce[6][3] = 3.0;
	ce[7][3] = 0.03;
	ce[8][3] = 0.05;
	ce[9][3] = 0.04;
	ce[10][3] = 0.2;
	ce[11][3] = 0.1;
	ce[12][3] = 0.3;

	ce[0][4] = 5.0;
	ce[1][4] = 4.0;
	ce[2][4] = 3.0;
	ce[3][4] = 2.0;
	ce[4][4] = 0.1;
	ce[5][4] = 0.4;
	ce[6][4] = 0.3;
	ce[7][4] = 0.05;
	ce[8][4] = 0.04;
	ce[9][4] = 0.03;
	ce[10][4] = 0.1;
	ce[11][4] = 0.3;
	ce[12][4] = 0.2;

	double dnxm1 = 1.0/((double)nx-1.0);
	double dnym1 = 1.0/((double)ny-1.0);
	double dnzm1 = 1.0/((double)nz-1.0);

	double tx1 = 1.0 / (dnxm1 * dnxm1);
	double tx2 = 1.0 / (2.0 * dnxm1);
	double tx3 = 1.0 / dnxm1;

	double ty1 = 1.0 / (dnym1 * dnym1);
	double ty2 = 1.0 / (2.0 * dnym1);
	double ty3 = 1.0 / dnym1;
 
	double tz1 = 1.0 / (dnzm1 * dnzm1);
	double tz2 = 1.0 / (2.0 * dnzm1);
	double tz3 = 1.0 / dnzm1;

	double dttx1 = dt*tx1;
	double dttx2 = dt*tx2;
	double dtty1 = dt*ty1;
	double dtty2 = dt*ty2;
	double dttz1 = dt*tz1;
	double dttz2 = dt*tz2;

	double c2dttx1 = 2.0*dttx1;
	double c2dtty1 = 2.0*dtty1;
	double c2dttz1 = 2.0*dttz1;

	double dtdssp = dt*dssp;

	double comz1  = dtdssp;
	double comz4  = 4.0*dtdssp;
	double comz5  = 5.0*dtdssp;
	double comz6  = 6.0*dtdssp;

	double c3c4tx3 = c3c4*tx3;
	double c3c4ty3 = c3c4*ty3;
	double c3c4tz3 = c3c4*tz3;

	double dx1tx1 = dx1*tx1;
	double dx2tx1 = dx2*tx1;
	double dx3tx1 = dx3*tx1;
	double dx4tx1 = dx4*tx1;
	double dx5tx1 = dx5*tx1;

	double dy1ty1 = dy1*ty1;
	double dy2ty1 = dy2*ty1;
	double dy3ty1 = dy3*ty1;
	double dy4ty1 = dy4*ty1;
	double dy5ty1 = dy5*ty1;

	double dz1tz1 = dz1*tz1;
	double dz2tz1 = dz2*tz1;
	double dz3tz1 = dz3*tz1;
	double dz4tz1 = dz4*tz1;
	double dz5tz1 = dz5*tz1;

	double xxcon1 = c3c4tx3*con43*tx3;
	double xxcon2 = c3c4tx3*tx3;
	double xxcon3 = c3c4tx3*conz1*tx3;
	double xxcon4 = c3c4tx3*con16*tx3;
	double xxcon5 = c3c4tx3*c1c5*tx3;

	double yycon1 = c3c4ty3*con43*ty3;
	double yycon2 = c3c4ty3*ty3;
	double yycon3 = c3c4ty3*conz1*ty3;
	double yycon4 = c3c4ty3*con16*ty3;
	double yycon5 = c3c4ty3*c1c5*ty3;

	double zzcon1 = c3c4tz3*con43*tz3;
	double zzcon2 = c3c4tz3*tz3;
	double zzcon3 = c3c4tz3*conz1*tz3;
	double zzcon4 = c3c4tz3*con16*tz3;
	double zzcon5 = c3c4tz3*c1c5*tz3;

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ce, &ce, 13*5*sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dnxm1, &dnxm1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dnym1, &dnym1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dnzm1, &dnzm1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx1, &tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx2, &tx2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx3, &tx3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty1, &ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty2, &ty2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty3, &ty3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz1, &tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz2, &tz2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz3, &tz3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dt, &dt, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttx1, &dttx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttx2, &dttx2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dtty1, &dtty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dtty2, &dtty2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttz1, &dttz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttz2, &dttz2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dttx1, &c2dttx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dtty1, &c2dtty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dttz1, &c2dttz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dtdssp, &dtdssp, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::comz1, &comz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::comz4, &comz4, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::comz5, &comz5, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::comz6, &comz6, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c3c4tx3, &c3c4tx3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c3c4ty3, &c3c4ty3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c3c4tz3, &c3c4tz3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dx1tx1, &dx1tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dx2tx1, &dx2tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dx3tx1, &dx3tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dx4tx1, &dx4tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dx5tx1, &dx5tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dy1ty1, &dy1ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dy2ty1, &dy2ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dy3ty1, &dy3ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dy4ty1, &dy4ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dy5ty1, &dy5ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dz1tz1, &dz1tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dz2tz1, &dz2tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dz3tz1, &dz3tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dz4tz1, &dz4tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dz5tz1, &dz5tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::xxcon1, &xxcon1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::xxcon2, &xxcon2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::xxcon3, &xxcon3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::xxcon4, &xxcon4, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::xxcon5, &xxcon5, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::yycon1, &yycon1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::yycon2, &yycon2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::yycon3, &yycon3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::yycon4, &yycon4, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::yycon5, &yycon5, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::zzcon1, &zzcon1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::zzcon2, &zzcon2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::zzcon3, &zzcon3, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::zzcon4, &zzcon4, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::zzcon5, &zzcon5, sizeof(double)));
}

void BT::allocate_device_memory() {
	int gridsize = nx*ny*nz;
	int facesize = max(max(nx*ny, nx*nz), ny*nz);

	HANDLE_ERROR(cudaMalloc((void **)&u, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&forcing, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rhs, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&lhs, 5*5*3*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rho_i, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&us, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&vs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&ws, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&qs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&square, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rmsbuf, 5*facesize*sizeof(double)));
}

void BT::free_device_memory() {
	HANDLE_ERROR(cudaFree(u));
	HANDLE_ERROR(cudaFree(forcing));
	HANDLE_ERROR(cudaFree(rhs));
	HANDLE_ERROR(cudaFree(lhs));
	HANDLE_ERROR(cudaFree(rho_i));
	HANDLE_ERROR(cudaFree(us));
	HANDLE_ERROR(cudaFree(vs));
	HANDLE_ERROR(cudaFree(ws));
	HANDLE_ERROR(cudaFree(qs));
	HANDLE_ERROR(cudaFree(square));
	HANDLE_ERROR(cudaFree(rmsbuf));
}

void BT::get_cuda_info() {
	int count;
	cudaDeviceProp prop;

	HANDLE_ERROR(cudaGetDeviceCount(&count));
	if (count == 0) {
		printf ("No CUDA devices found.\n");
		exit(EXIT_FAILURE);
	}

	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	strncpy (CUDAname, prop.name, 256);
	CUDAmp = prop.multiProcessorCount;
	CUDAclock = prop.clockRate;
	CUDAmem = prop.totalGlobalMem;
	CUDAmemclock = prop.memoryClockRate;
	CUDAl2cache = prop.l2CacheSize;
}
