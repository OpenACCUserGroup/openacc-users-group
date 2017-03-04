#include <stdio.h>
#include "main.h"

namespace gpu_mod {
__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
__constant__ double bt, dt, dtdssp;
__constant__ double dnxm1, dnym1, dnzm1;
__constant__ double dttx1, dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, c2dtty1, c2dttz1;
__constant__ double comz1, comz4, comz5, comz6, c3c4tx3, c3c4ty3, c3c4tz3;
__constant__ double xxcon1, xxcon2, xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1, dx4tx1, dx5tx1;
__constant__ double yycon1, yycon2, yycon3, yycon4, yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1;
__constant__ double zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, dz2tz1, dz3tz1, dz4tz1, dz5tz1;
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

void SP::adi (bool singlestep) {

	for (int i = 0; i < t_last; i++) timers->timer_clear(i);
	HANDLE_ERROR(cudaDeviceSynchronize());
	timers->timer_start(0);

	int itmax = singlestep ? 1 : niter;
	for (int step = 1; step <= itmax; step++) {
		if (step % 20 == 0 || step == 1 && !singlestep)
			printf(" Time step %4d\n", step);

		compute_rhs();
		txinvr();
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
// 	addition of update to the vector u
//---------------------------------------------------------------------
__global__ static void add_kernel (double *u, const double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k, m;

	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;
	m = threadIdx.y;

	u(m,i,j,k) += rhs(m,i,j,k);
}

void SP::add () {
	dim3 grid(ny-2,nz-2);
	dim3 block(nx-2,5);

	START_TIMER(t_add);
	add_kernel<<<grid,block>>>(u, rhs, nx, ny, nz);
	STOP_TIMER(t_add);
}

__global__ static void compute_rhs_kernel_1 (double *rho_i, double *us, double *vs, double *ws, double *speed, double *qs, double *square, const double *u, const int nx, const int ny, const int nz) {
	int i, j, k;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	//---------------------------------------------------------------------
	//      compute the reciprocal of density, and the kinetic energy, 
	//      and the speed of sound. 
	//---------------------------------------------------------------------
	double rho_inv = 1.0/u(0,i,j,k);
	double square_ijk;
	rho_i(i,j,k) = rho_inv;
	us(i,j,k) = u(1,i,j,k) * rho_inv;
	vs(i,j,k) = u(2,i,j,k) * rho_inv;
	ws(i,j,k) = u(3,i,j,k) * rho_inv;
	square(i,j,k) = square_ijk = 0.5*(u(1,i,j,k)*u(1,i,j,k) + u(2,i,j,k)*u(2,i,j,k) + u(3,i,j,k)*u(3,i,j,k)) * rho_inv;
	qs(i,j,k) = square_ijk * rho_inv;
	//---------------------------------------------------------------------
	//               (don't need speed and ainx until the lhs computation)
	//---------------------------------------------------------------------
	speed(i,j,k) = sqrt(c1c2*rho_inv*(u(4,i,j,k) - square_ijk));
}

__global__ static void compute_rhs_kernel_2 (const double *rho_i, const double *us, const double *vs, const double *ws, const double *qs, const double *square, double *rhs, const double *forcing, const double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	k = blockIdx.y;
	j = blockIdx.x;
	i = threadIdx.x;
	double rtmp[5];

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	// copy the exact forcing term to the right hand side;  because 
	// this forcing term is known, we can store it on the whole zone
	// including the boundary                   
	//---------------------------------------------------------------------
	for (m = 0; m < 5; m++) rtmp[m] = forcing(m,i,j,k);
	
	//---------------------------------------------------------------------
	//      compute xi-direction fluxes 
	//---------------------------------------------------------------------
	if (k >= 1 && k < nz-1 && j >= 1 && j < ny-1 && i >= 1 && i < nx-1) {
		double uijk = us(i,j,k);
		double up1 = us(i+1,j,k);
		double um1 = us(i-1,j,k);
				
		rtmp[0] = rtmp[0] + dx1tx1*(u(0,i+1,j,k) - 2.0*u(0,i,j,k) + u(0,i-1,j,k)) - tx2*(u(1,i+1,j,k)-u(1,i-1,j,k));
		rtmp[1] = rtmp[1] + dx2tx1*(u(1,i+1,j,k) - 2.0*u(1,i,j,k) + u(1,i-1,j,k)) + xxcon2*con43*(up1-2.0*uijk+um1) - tx2*(u(1,i+1,j,k)*up1 - u(1,i-1,j,k)*um1 + (u(4,i+1,j,k)-square(i+1,j,k)-u(4,i-1,j,k)+square(i-1,j,k))*c2);
		rtmp[2] = rtmp[2] + dx3tx1*(u(2,i+1,j,k) - 2.0*u(2,i,j,k) + u(2,i-1,j,k)) + xxcon2*(vs(i+1,j,k)-2.0*vs(i,j,k)+vs(i-1,j,k)) - tx2*(u(2,i+1,j,k)*up1 - u(2,i-1,j,k)*um1);
		rtmp[3] = rtmp[3] + dx4tx1*(u(3,i+1,j,k) - 2.0*u(3,i,j,k) + u(3,i-1,j,k)) + xxcon2*(ws(i+1,j,k)-2.0*ws(i,j,k)+ws(i-1,j,k)) - tx2*(u(3,i+1,j,k)*up1 - u(3,i-1,j,k)*um1);
		rtmp[4] = rtmp[4] + dx5tx1*(u(4,i+1,j,k) - 2.0*u(4,i,j,k) + u(4,i-1,j,k)) + xxcon3*(qs(i+1,j,k)-2.0*qs(i,j,k)+qs(i-1,j,k))+ xxcon4*(up1*up1-2.0*uijk*uijk+um1*um1) +
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
			for (m = 0; m < 5; m++)	rtmp[m] = rtmp[m] - dssp*(5.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
		} else if (k == 2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
		} else if (k >= 3 && k < nz-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1)+u(m,i,j,k+2));
		} else if (k == nz-3) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+6.0*u(m,i,j,k)-4.0*u(m,i,j,k+1));
		} else if (k == nz-2) {
			for (m = 0; m < 5; m++) rtmp[m] = rtmp[m] - dssp*(u(m,i,j,k-2)-4.0*u(m,i,j,k-1)+5.0*u(m,i,j,k));
		}

		for (m = 0; m < 5; m++) rtmp[m] *= dt;
	}

	for (m = 0; m < 5; m++) rhs(m,i,j,k) = rtmp[m];
}

void SP::compute_rhs () {
	dim3 grid1(ny,nz);

	START_TIMER(t_rhs);
	compute_rhs_kernel_1<<<grid1,nx>>>(rho_i, us, vs, ws, speed, qs, square, u, nx, ny, nz);

	START_TIMER(t_rhsx);
	compute_rhs_kernel_2<<<grid1,nx>>>(rho_i, us, vs, ws, qs, square, rhs, forcing, u, nx, ny, nz);
	STOP_TIMER(t_rhsx);

	STOP_TIMER(t_rhs);
}

__global__ static void txinvr_kernel (const double *rho_i, const double *us, const double *vs, const double *ws, const double *speed, const double *qs, double *rhs, const int nx, const int ny, const int nz) {
	int i, j, k;

	k = blockIdx.y+1;
	j = blockIdx.x+1;
	i = threadIdx.x+1;

	using namespace gpu_mod;

	double ru1 = rho_i(i,j,k);
	double uu = us(i,j,k);
	double vv = vs(i,j,k);
	double ww = ws(i,j,k);
	double ac = speed(i,j,k);
	double ac2inv = 1.0/( ac*ac );

	double r1 = rhs(0,i,j,k);
	double r2 = rhs(1,i,j,k);
	double r3 = rhs(2,i,j,k);
	double r4 = rhs(3,i,j,k);
	double r5 = rhs(4,i,j,k);

	double t1 = c2*ac2inv*(qs(i,j,k)*r1 - uu*r2  - vv*r3 - ww*r4 + r5);
	double t2 = bt * ru1 * ( uu * r1 - r2 );
	double t3 = ( bt * ru1 * ac ) * t1;

	rhs(0,i,j,k) = r1 - t1;
	rhs(1,i,j,k) = -ru1*(ww*r1-r4);
	rhs(2,i,j,k) = ru1*(vv*r1-r3);
	rhs(3,i,j,k) = -t2+t3;
	rhs(4,i,j,k) = t2+t3;
}

void SP::txinvr () {
	dim3 grid (ny-2,nz-2);

	START_TIMER(t_txinvr);
	txinvr_kernel<<<grid,nx-2>>> (rho_i, us, vs, ws, speed, qs, rhs, nx, ny, nz);
	STOP_TIMER(t_txinvr);
}

//---------------------------------------------------------------------
// Computes the left hand side for the three x-factors  
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3)))]
#define lhsp(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m+4)))]
#define lhsm(m,i,j,k) lhs[(j-1)+(ny-2)*((k-1)+(nz-2)*((i)+nx*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(j)+ny*((k)+nz*((i)+nx*(m)))]
__global__ static void x_solve_kernel (const double *rho_i, const double *us, const double *speed, double *rhs, double *lhs, double *rhstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhon[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	j = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || j >= ny-1) return;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three x-factors  
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_lhs[0][0] = lhsp(0,0,j,k) = 0.0;
	_lhs[0][1] = lhsp(1,0,j,k) = 0.0;
	_lhs[0][2] = lhsp(2,0,j,k) = 1.0;
	_lhs[0][3] = lhsp(3,0,j,k) = 0.0;
	_lhs[0][4] = lhsp(4,0,j,k) = 0.0;

	//---------------------------------------------------------------------
	// first fill the lhs for the u-eigenvalue                          
	//---------------------------------------------------------------------
	for (i = 0; i < 3; i++) {
		fac1 = c3c4*rho_i(i,j,k);
		rhon[i] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1);
		cv[i] = us(i,j,k);
	}
	_lhs[1][0] = 0.0;
	_lhs[1][1] = - dttx2 * cv[0] - dttx1 * rhon[0];
	_lhs[1][2] = 1.0 + c2dttx1 * rhon[1];
	_lhs[1][3] = dttx2 * cv[2] - dttx1 * rhon[2];
	_lhs[1][4] = 0.0;
	_lhs[1][2] += comz5;
	_lhs[1][3] -= comz4;
	_lhs[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,1,j,k) = _lhs[1][m];
	rhon[0] = rhon[1]; rhon[1] = rhon[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rhs[0][m] = rhs(m,0,j,k);
		_rhs[1][m] = rhs(m,1,j,k);
	}

	//---------------------------------------------------------------------
	//      perform the Thomas algorithm; first, FORWARD ELIMINATION     
	//---------------------------------------------------------------------
	for (i = 0; i < nx-2; i++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (i+2 == nx-1) {
			_lhs[2][0] = lhsp(0,i+2,j,k) = 0.0;
			_lhs[2][1] = lhsp(1,i+2,j,k) = 0.0;
			_lhs[2][2] = lhsp(2,i+2,j,k) = 1.0;
			_lhs[2][3] = lhsp(3,i+2,j,k) = 0.0;
			_lhs[2][4] = lhsp(4,i+2,j,k) = 0.0;
		} else {
			fac1 = c3c4*rho_i(i+3,j,k);
			rhon[2] = max(max(max(dx2+con43*fac1, dx5+c1c5*fac1), dxmax+fac1), dx1);
			cv[2] = us(i+3,j,k);
			_lhs[2][0] = 0.0;
			_lhs[2][1] = - dttx2 * cv[0] - dttx1 * rhon[0];
			_lhs[2][2] = 1.0 + c2dttx1 * rhon[1];
			_lhs[2][3] = dttx2 * cv[2] - dttx1 * rhon[2];
			_lhs[2][4] = 0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                                  
			//---------------------------------------------------------------------
			if (i+2 == 2) {
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (i+2 >= 3 && i+2 < nx-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (i+2 == nx-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
			} else if (i+2 == nx-2) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i+2,j,k) = _lhs[2][m];
			rhon[0] = rhon[1]; rhon[1] = rhon[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m,i+2,j,k);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
		_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
		_lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
		_lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _lhs[0][3];
		lhs(4,i,j,k) = _lhs[0][4];
		for (m = 0; m < 5; m++) {
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
	}

	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	i = nx-2;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
	_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
	for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
	//---------------------------------------------------------------------
	//            scale the last row immediately 
	//---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2];
	for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;
	lhs(3,nx-2,j,k) = _lhs[0][3];
	lhs(4,nx-2,j,k) = _lhs[0][4];

	//---------------------------------------------------------------------
	//      subsequently, fill the other factors (u+c), (u-c) 
	//---------------------------------------------------------------------
	for (i = 0; i < 3; i++) cv[i] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_lhsp[0][m] = _lhs[0][m] = lhsp(m,0,j,k);
		_lhsp[1][m] = _lhs[1][m] = lhsp(m,1,j,k);
	}
	_lhsp[1][1] -= dttx2 * cv[0];
	_lhsp[1][3] += dttx2 * cv[2];
	_lhs[1][1] += dttx2 * cv[0];
	_lhs[1][3] -= dttx2 * cv[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	_rhs[0][3] = rhs(3,0,j,k);
	_rhs[0][4] = rhs(4,0,j,k);
	_rhs[1][3] = rhs(3,1,j,k);
	_rhs[1][4] = rhs(4,1,j,k);
	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors               
	//---------------------------------------------------------------------
	for (i = 0; i < nx-2; i++) {
		//---------------------------------------------------------------------
		//      first, fill the other factors (u+c), (u-c) 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) {
			_lhsp[2][m] = _lhs[2][m] = lhsp(m,i+2,j,k);
		}
		_rhs[2][3] = rhs(3,i+2,j,k);
		_rhs[2][4] = rhs(4,i+2,j,k);

		if (i+2 < nx-1) {
			cv[2] = speed(i+3,j,k);
			_lhsp[2][1] -= dttx2 * cv[0];
			_lhsp[2][3] += dttx2 * cv[2];
			_lhs[2][1] += dttx2 * cv[0];
			_lhs[2][3] -= dttx2 * cv[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		m = 3;
		fac1 = 1.0/_lhsp[0][2];
		_lhsp[0][3] *= fac1;
		_lhsp[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3];
		_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4];
		_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m];
		_lhsp[2][1] -= _lhsp[2][0]*_lhsp[0][3];
		_lhsp[2][2] -= _lhsp[2][0]*_lhsp[0][4];
		_rhs[2][m] -= _lhsp[2][0]*_rhs[0][m];

		m = 4;
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1]*_lhs[0][3];
		_lhs[1][3] -= _lhs[1][1]*_lhs[0][4];
		_rhs[1][m] -= _lhs[1][1]*_rhs[0][m];
		_lhs[2][1] -= _lhs[2][0]*_lhs[0][3];
		_lhs[2][2] -= _lhs[2][0]*_lhs[0][4];
		_rhs[2][m] -= _lhs[2][0]*_rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lhsp[0][m];
			lhsm(m,i,j,k) = _lhs[0][m];
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lhsp[0][m] = _lhsp[1][m];
			_lhsp[1][m] = _lhsp[2][m];
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	i = nx-2;
	m = 3;
	fac1 = 1.0/_lhsp[0][2];
	_lhsp[0][3] *= fac1;
	_lhsp[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhsp[1][2] -= _lhsp[1][1]*_lhsp[0][3];
	_lhsp[1][3] -= _lhsp[1][1]*_lhsp[0][4];
	_rhs[1][m] -= _lhsp[1][1]*_rhs[0][m];

	m = 4;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1]*_lhs[0][3];
	_lhs[1][3] -= _lhs[1][1]*_lhs[0][4];
	_rhs[1][m] -= _lhs[1][1]*_rhs[0][m];

	//---------------------------------------------------------------------
	//               Scale the last row immediately
	//---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2];
	_rhs[1][4] /= _lhs[1][2];

	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3,nx-2,j,k)*_rhs[1][m];
	_rhs[0][3] -= _lhsp[0][3]*_rhs[1][3];
	_rhs[0][4] -= _lhs[0][3]*_rhs[1][4];
	for (m = 0; m < 5; m++) {
		_rhs[2][m] = _rhs[1][m];
		_rhs[1][m] = _rhs[0][m];
	}

	for (i = nx-3; i >= 0; i--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rhs[1][m] - lhs(4,i,j,k)*_rhs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rhs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rhs[1][3] - lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rhs[1][4] - lhsm(4,i,j,k)*_rhs[2][4];

		if (i+2 < nx-1) {
			//---------------------------------------------------------------------
			//      Do the block-diagonal inversion          
			//---------------------------------------------------------------------
				double r1 = _rhs[2][0];
				double r2 = _rhs[2][1];
				double r3 = _rhs[2][2];
				double r4 = _rhs[2][3];
				double r5 = _rhs[2][4];
				double t1 = bt * r3;
				double t2 = 0.5 * (r4+r5);

				_rhs[2][0] = -r2;
				_rhs[2][1] =  r1;
				_rhs[2][2] = bt * ( r4 - r5 );
				_rhs[2][3] = -t1 + t2;
				_rhs[2][4] =  t1 + t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i+2,j,k) = _rhs[2][m];
			_rhs[2][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//      Do the block-diagonal inversion          
	//---------------------------------------------------------------------
	double t1 = bt * _rhs[2][2];
	double t2 = 0.5 * (_rhs[2][3]+_rhs[2][4]);
	rhs(0,1,j,k) = -_rhs[2][1];
	rhs(1,1,j,k) =  _rhs[2][0];
	rhs(2,1,j,k) = bt * ( _rhs[2][3] - _rhs[2][4] );
	rhs(3,1,j,k) = -t1 + t2;
	rhs(4,1,j,k) =  t1 + t2;

	for (m = 0; m < 5; m++) rhs(m,0,j,k) = _rhs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp

void SP::x_solve () {
	int yblock = min(SOLVE_BLOCK,ny);
	int ygrid = (ny+yblock-1)/yblock;
	int zblock = min(SOLVE_BLOCK/yblock,nz);
	int zgrid = (nz+zblock-1)/zblock;
	dim3 grid(zgrid,ygrid), block(zblock,yblock);

	START_TIMER(t_xsolve);
	x_solve_kernel<<<grid,block>>>(rho_i, us, speed, rhs, lhs, rhstmp, nx, ny, nz);
	STOP_TIMER(t_xsolve);
}

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((k-1)+(nz-2)*((j)+ny*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(i)+nx*((k)+nz*((j)+ny*(m)))]
__global__ static void y_solve_kernel (const double *rho_i, const double *vs, const double *speed, double *rhs, double *lhs, double *rhstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhoq[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	k = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (k >= nz-1 || i >= nx-1) return;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three y-factors   
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_lhs[0][0] = lhsp(0,i,0,k) = 0.0;
	_lhs[0][1] = lhsp(1,i,0,k) = 0.0;
	_lhs[0][2] = lhsp(2,i,0,k) = 1.0;
	_lhs[0][3] = lhsp(3,i,0,k) = 0.0;
	_lhs[0][4] = lhsp(4,i,0,k) = 0.0;

	//---------------------------------------------------------------------
	//      first fill the lhs for the u-eigenvalue         
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) {
		fac1 = c3c4*rho_i(i,j,k);
		rhoq[j] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), dy1);
		cv[j] = vs(i,j,k);
	}
	_lhs[1][0] =  0.0;
	_lhs[1][1] = -dtty2*cv[0]-dtty1 * rhoq[0];
	_lhs[1][2] =  1.0 + c2dtty1 * rhoq[1];
	_lhs[1][3] =  dtty2*cv[2]-dtty1 * rhoq[2];
	_lhs[1][4] =  0.0;
	_lhs[1][2] += comz5;
	_lhs[1][3] -= comz4;
	_lhs[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,i,1,k) = _lhs[1][m];
	rhoq[0] = rhoq[1]; rhoq[1] = rhoq[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rhs[0][m] = rhs(m,i,0,k);
		_rhs[1][m] = rhs(m,i,1,k);
	}

	//---------------------------------------------------------------------
	//                          FORWARD ELIMINATION  
	//---------------------------------------------------------------------
	for (j = 0; j < ny-2; j++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (j+2 == ny-1) {
			_lhs[2][0] = lhsp(0,i,j+2,k) = 0.0;
			_lhs[2][1] = lhsp(1,i,j+2,k) = 0.0;
			_lhs[2][2] = lhsp(2,i,j+2,k) = 1.0;
			_lhs[2][3] = lhsp(3,i,j+2,k) = 0.0;
			_lhs[2][4] = lhsp(4,i,j+2,k) = 0.0;
		} else {
			fac1 = c3c4*rho_i(i,j+3,k);
			rhoq[2] = max(max(max(dy3+con43*fac1, dy5+c1c5*fac1), dymax+fac1), dy1);
			cv[2] = vs(i,j+3,k);
			_lhs[2][0] =  0.0;
			_lhs[2][1] = -dtty2*cv[0]-dtty1 * rhoq[0];
			_lhs[2][2] =  1.0 + c2dtty1 * rhoq[1];
			_lhs[2][3] =  dtty2*cv[2]-dtty1 * rhoq[2];
			_lhs[2][4] =  0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                             
			//---------------------------------------------------------------------
			if (j+2 == 2) {
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (j+2 >= 3 && j+2 < ny-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (j+2 == ny-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
			} else if (j+2 == ny-2) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i,j+2,k) = _lhs[2][m];
			rhoq[0] = rhoq[1]; rhoq[1] = rhoq[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m,i,j+2,k);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
		_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
		_lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
		_lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _lhs[0][3];
		lhs(4,i,j,k) = _lhs[0][4];
		for (m = 0; m < 5; m++) {
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	j = ny-2;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
	_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
	for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
	//---------------------------------------------------------------------
	//            scale the last row immediately 
	//---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2];
	for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;
	lhs(3,i,ny-2,k) = _lhs[0][3];
	lhs(4,i,ny-2,k) = _lhs[0][4];

	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors                 
	//---------------------------------------------------------------------
	for (j = 0; j < 3; j++) cv[j] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_lhsp[0][m] = _lhs[0][m] = lhsp(m,i,0,k);
		_lhsp[1][m] = _lhs[1][m] = lhsp(m,i,1,k);
	}
	_lhsp[1][1] -= dtty2*cv[0];
	_lhsp[1][3] += dtty2*cv[2];
	_lhs[1][1] += dtty2*cv[0];
	_lhs[1][3] -= dtty2*cv[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	_rhs[0][3] = rhs(3,i,0,k);
	_rhs[0][4] = rhs(4,i,0,k);
	_rhs[1][3] = rhs(3,i,1,k);
	_rhs[1][4] = rhs(4,i,1,k);
	for (j = 0; j < ny-2; j++) {
		for (m = 0; m < 5; m++) {
			_lhsp[2][m] = _lhs[2][m] = lhsp(m,i,j+2,k);
		}
		_rhs[2][3] = rhs(3,i,j+2,k);
		_rhs[2][4] = rhs(4,i,j+2,k);
		if (j+2 < ny-1) {
			cv[2] = speed(i,j+3,k);
			_lhsp[2][1] -= dtty2*cv[0];
			_lhsp[2][3] += dtty2*cv[2];
			_lhs[2][1] += dtty2*cv[0];
			_lhs[2][3] -= dtty2*cv[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		fac1 = 1.0/_lhsp[0][2];
		m = 3;
		_lhsp[0][3] *= fac1;
		_lhsp[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
		_lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
		_rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
		_lhsp[2][1] -= _lhsp[2][0] * _lhsp[0][3];
		_lhsp[2][2] -= _lhsp[2][0] * _lhsp[0][4];
		_rhs[2][m] -= _lhsp[2][0] * _rhs[0][m];

		m = 4;
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
		_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
		_rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
		_lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
		_lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
		_rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lhsp[0][m];
			lhsm(m,i,j,k) = _lhs[0][m];
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lhsp[0][m] = _lhsp[1][m];
			_lhsp[1][m] = _lhsp[2][m];
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	j = ny-2;
	m = 3;
	fac1 = 1.0/_lhsp[0][2];
	_lhsp[0][3] *= fac1;
	_lhsp[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
	_lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
	_rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];

	m = 4;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
	_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
	_rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
	//---------------------------------------------------------------------
	//               Scale the last row immediately 
	//---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2];
	_rhs[1][4] /= _lhs[1][2];

	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3,i,ny-2,k) * _rhs[1][m];
	_rhs[0][3] -= _lhsp[0][3] * _rhs[1][3];
	_rhs[0][4] -= _lhs[0][3] * _rhs[1][4];
	for (m = 0; m < 5; m++) {
		_rhs[2][m] = _rhs[1][m];
		_rhs[1][m] = _rhs[0][m];
	}
	for (j = ny-3; j >= 0; j--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rhs[1][m] - lhs(4,i,j,k)*_rhs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rhs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rhs[1][3] - lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rhs[1][4] - lhsm(4,i,j,k)*_rhs[2][4];
	
		if (j+2 < ny-1) {
			//---------------------------------------------------------------------
			//   block-diagonal matrix-vector multiplication                       
			//---------------------------------------------------------------------
			double r1 = _rhs[2][0];
			double r2 = _rhs[2][1];
			double r3 = _rhs[2][2];
			double r4 = _rhs[2][3];
			double r5 = _rhs[2][4];

			double t1 = bt * r1;
			double t2 = 0.5 * ( r4 + r5 );

			_rhs[2][0] =  bt * ( r4 - r5 );
			_rhs[2][1] = -r3;
			_rhs[2][2] =  r2;
			_rhs[2][3] = -t1 + t2;
			_rhs[2][4] =  t1 + t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i,j+2,k) = _rhs[2][m];
			_rhs[2][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//   block-diagonal matrix-vector multiplication                       
	//---------------------------------------------------------------------
	double t1 = bt * _rhs[2][0];
	double t2 = 0.5 * ( _rhs[2][3] + _rhs[2][4] );
	rhs(0,i,1,k) =  bt * ( _rhs[2][3] - _rhs[2][4] );
	rhs(1,i,1,k) = -_rhs[2][2];
	rhs(2,i,1,k) =  _rhs[2][1];
	rhs(3,i,1,k) = -t1 + t2;
	rhs(4,i,1,k) =  t1 + t2;

	for (m = 0; m < 5; m++) rhs(m,i,0,k) = _rhs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp

void SP::y_solve () {
	int xblock = min(SOLVE_BLOCK,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int zblock = min(SOLVE_BLOCK/xblock,nz);
	int zgrid = (nz+zblock-1)/zblock;
	dim3 grid(zgrid,xgrid), block(zblock,xblock);

	START_TIMER(t_ysolve);
	y_solve_kernel<<<grid,block>>>(rho_i, vs, speed, rhs, lhs, rhstmp, nx, ny, nz);
	STOP_TIMER(t_ysolve);
}

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
#define lhs(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3)))]
#define lhsp(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m+4)))]
#define lhsm(m,i,j,k) lhs[(i-1)+(nx-2)*((j-1)+(ny-2)*((k)+nz*(m-3+2)))]
#define rtmp(m,i,j,k) rhstmp[(i)+nx*((j)+ny*((k)+nz*(m)))]
__global__ static void z_solve_kernel (const double *rho_i, const double *us, const double *vs, const double *ws, const double *speed, const double *qs, const double *u, double *rhs, double *lhs, double *rhstmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double rhos[3], cv[3], _lhs[3][5], _lhsp[3][5], _rhs[3][5], fac1;

	j = blockIdx.x*blockDim.x+threadIdx.x+1;
	i = blockIdx.y*blockDim.y+threadIdx.y+1;
	if (j >= ny-1 || i >= nx-1) return;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	// Computes the left hand side for the three z-factors   
	//---------------------------------------------------------------------
	//---------------------------------------------------------------------
	//     zap the whole left hand side for starters
	//---------------------------------------------------------------------
	_lhs[0][0] = lhsp(0,i,j,0) = 0.0;
	_lhs[0][1] = lhsp(1,i,j,0) = 0.0;
	_lhs[0][2] = lhsp(2,i,j,0) = 1.0;
	_lhs[0][3] = lhsp(3,i,j,0) = 0.0;
	_lhs[0][4] = lhsp(4,i,j,0) = 0.0;

	//---------------------------------------------------------------------
	// first fill the lhs for the u-eigenvalue                          
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) {
		fac1 = c3c4*rho_i(i,j,k);
		rhos[k] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), dz1);
		cv[k] = ws(i,j,k);
	}
	_lhs[1][0] =  0.0;
	_lhs[1][1] = -dttz2*cv[0] - dttz1*rhos[0];
	_lhs[1][2] =  1.0 + c2dttz1 * rhos[1];
	_lhs[1][3] =  dttz2*cv[2] - dttz1*rhos[2];
	_lhs[1][4]=  0.0;
	_lhs[1][2] += comz5;
	_lhs[1][3] -= comz4;
	_lhs[1][4] += comz1;
	for (m = 0; m < 5; m++) lhsp(m,i,j,1) = _lhs[1][m];
	rhos[0] = rhos[1]; rhos[1] = rhos[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	for (m = 0; m < 3; m++) {
		_rhs[0][m] = rhs(m,i,j,0);
		_rhs[1][m] = rhs(m,i,j,1);
	}

	//---------------------------------------------------------------------
	//                          FORWARD ELIMINATION  
	//---------------------------------------------------------------------
	for (k = 0; k < nz-2; k++) {
		//---------------------------------------------------------------------
		// first fill the lhs for the u-eigenvalue                          
		//---------------------------------------------------------------------
		if (k+2 == nz-1) {
			_lhs[2][0] = lhsp(0,i,j,k+2) = 0.0;
			_lhs[2][1] = lhsp(1,i,j,k+2) = 0.0;
			_lhs[2][2] = lhsp(2,i,j,k+2) = 1.0;
			_lhs[2][3] = lhsp(3,i,j,k+2) = 0.0;
			_lhs[2][4] = lhsp(4,i,j,k+2) = 0.0;
		} else {
			fac1 = c3c4*rho_i(i,j,k+3);
			rhos[2] = max(max(max(dz4+con43*fac1, dz5+c1c5*fac1), dzmax+fac1), dz1);
			cv[2] = ws(i,j,k+3);
			_lhs[2][0] =  0.0;
			_lhs[2][1] = -dttz2*cv[0] - dttz1*rhos[0];
			_lhs[2][2] =  1.0 + c2dttz1 * rhos[1];
			_lhs[2][3] =  dttz2*cv[2] - dttz1*rhos[2];
			_lhs[2][4] =  0.0;
			//---------------------------------------------------------------------
			//      add fourth order dissipation                                  
			//---------------------------------------------------------------------
			if (k+2 == 2) {
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (k+2 >= 3 && k+2 < nz-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
				_lhs[2][4] += comz1;
			} else if (k+2 == nz-3) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz6;
				_lhs[2][3] -= comz4;
			} else if (k+2 == nz-2) {
				_lhs[2][0] += comz1;
				_lhs[2][1] -= comz4;
				_lhs[2][2] += comz5;
			}

			//---------------------------------------------------------------------
			//      store computed lhs for later reuse
			//---------------------------------------------------------------------
			for (m = 0; m < 5; m++) lhsp(m,i,j,k+2) = _lhs[2][m];
			rhos[0] = rhos[1]; rhos[1] = rhos[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		//---------------------------------------------------------------------
		//      load rhs values for current iteration
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[2][m] = rhs(m,i,j,k+2);

		//---------------------------------------------------------------------
		//      perform current iteration
		//---------------------------------------------------------------------
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
		_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
		_lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
		_lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
		for (m = 0; m < 3; m++) _rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		lhs(3,i,j,k) = _lhs[0][3];
		lhs(4,i,j,k) = _lhs[0][4];
		for (m = 0; m < 5; m++) {
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
		for (m = 0; m < 3; m++) {
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//      The last two rows in this zone are a bit different, 
	//      since they do not have two more rows available for the
	//      elimination of off-diagonal entries
	//---------------------------------------------------------------------
	k = nz-2;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	for (m = 0; m < 3; m++) _rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
	_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
	for (m = 0; m < 3; m++) _rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
	//---------------------------------------------------------------------
	//               scale the last row immediately
	//---------------------------------------------------------------------
	fac1 = 1.0/_lhs[1][2];
	for (m = 0; m < 3; m++) _rhs[1][m] *= fac1;
	lhs(3,i,j,k) = _lhs[0][3];
	lhs(4,i,j,k) = _lhs[0][4];

	//---------------------------------------------------------------------
	//      subsequently, fill the other factors (u+c), (u-c) 
	//---------------------------------------------------------------------
	for (k = 0; k < 3; k++) cv[k] = speed(i,j,k);
	for (m = 0; m < 5; m++) {
		_lhsp[0][m] = _lhs[0][m] = lhsp(m,i,j,0);
		_lhsp[1][m] = _lhs[1][m] = lhsp(m,i,j,1);
	}
	_lhsp[1][1] -= dttz2*cv[0];
	_lhsp[1][3] += dttz2*cv[2];
	_lhs[1][1] += dttz2*cv[0];
	_lhs[1][3] -= dttz2*cv[2];
	cv[0] = cv[1]; cv[1] = cv[2];
	_rhs[0][3] = rhs(3,i,j,0);
	_rhs[0][4] = rhs(4,i,j,0);
	_rhs[1][3] = rhs(3,i,j,1);
	_rhs[1][4] = rhs(4,i,j,1);
	//---------------------------------------------------------------------
	//      do the u+c and the u-c factors               
	//---------------------------------------------------------------------
	for (k = 0; k < nz-2; k++) {
		//---------------------------------------------------------------------
		//      first, fill the other factors (u+c), (u-c) 
		//---------------------------------------------------------------------
		for (m = 0; m < 5; m++) {
			_lhsp[2][m] = _lhs[2][m] = lhsp(m,i,j,k+2);
		}
		_rhs[2][3] = rhs(3,i,j,k+2);
		_rhs[2][4] = rhs(4,i,j,k+2);
		if (k+2 < nz-1) {
			cv[2] = speed(i,j,k+3);
			_lhsp[2][1] -= dttz2*cv[0];
			_lhsp[2][3] += dttz2*cv[2];
			_lhs[2][1] += dttz2*cv[0];
			_lhs[2][3] -= dttz2*cv[2];
			cv[0] = cv[1]; cv[1] = cv[2];
		}

		m = 3;
		fac1 = 1.0/_lhsp[0][2];
		_lhsp[0][3] *= fac1;
		_lhsp[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
		_lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
		_rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];
		_lhsp[2][1] -= _lhsp[2][0] * _lhsp[0][3];
		_lhsp[2][2] -= _lhsp[2][0] * _lhsp[0][4];
		_rhs[2][m] -= _lhsp[2][0] * _rhs[0][m];

		m = 4;
		fac1 = 1.0/_lhs[0][2];
		_lhs[0][3] *= fac1;
		_lhs[0][4] *= fac1;
		_rhs[0][m] *= fac1;
		_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
		_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
		_rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
		_lhs[2][1] -= _lhs[2][0] * _lhs[0][3];
		_lhs[2][2] -= _lhs[2][0] * _lhs[0][4];
		_rhs[2][m] -= _lhs[2][0] * _rhs[0][m];

		//---------------------------------------------------------------------
		//      store computed lhs and prepare data for next iteration
		//	rhs is stored in a temp array such that write accesses are coalesced
		//---------------------------------------------------------------------
		for (m = 3; m < 5; m++) {
			lhsp(m,i,j,k) = _lhsp[0][m];
			lhsm(m,i,j,k) = _lhs[0][m];
			rtmp(m,i,j,k) = _rhs[0][m];
			_rhs[0][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[2][m];
		}
		for (m = 0; m < 5; m++) {
			_lhsp[0][m] = _lhsp[1][m];
			_lhsp[1][m] = _lhsp[2][m];
			_lhs[0][m] = _lhs[1][m];
			_lhs[1][m] = _lhs[2][m];
		}
	}
	//---------------------------------------------------------------------
	//         And again the last two rows separately
	//---------------------------------------------------------------------
	k = nz-2;
	m = 3;
	fac1 = 1.0/_lhsp[0][2];
	_lhsp[0][3] *= fac1;
	_lhsp[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhsp[1][2] -= _lhsp[1][1] * _lhsp[0][3];
	_lhsp[1][3] -= _lhsp[1][1] * _lhsp[0][4];
	_rhs[1][m] -= _lhsp[1][1] * _rhs[0][m];

	m = 4;
	fac1 = 1.0/_lhs[0][2];
	_lhs[0][3] *= fac1;
	_lhs[0][4] *= fac1;
	_rhs[0][m] *= fac1;
	_lhs[1][2] -= _lhs[1][1] * _lhs[0][3];
	_lhs[1][3] -= _lhs[1][1] * _lhs[0][4];
	_rhs[1][m] -= _lhs[1][1] * _rhs[0][m];
	//---------------------------------------------------------------------
	//               Scale the last row immediately (some of this is overkill
	//               if this is the last cell)
	//---------------------------------------------------------------------
	_rhs[1][3] /= _lhsp[1][2];
	_rhs[1][4] /= _lhs[1][2];
		
	//---------------------------------------------------------------------
	//                         BACKSUBSTITUTION 
	//---------------------------------------------------------------------
	for (m = 0; m < 3; m++) _rhs[0][m] -= lhs(3,i,j,nz-2) * _rhs[1][m];
	_rhs[0][3] -= _lhsp[0][3] * _rhs[1][3];
	_rhs[0][4] -= _lhs[0][3] * _rhs[1][4];
	for (m = 0; m < 5; m++) {
		_rhs[2][m] = _rhs[1][m];
		_rhs[1][m] = _rhs[0][m];
	}
	
	for (k = nz-3; k >= 0; k--) {
		//---------------------------------------------------------------------
		//      The first three factors
		//---------------------------------------------------------------------
		for (m = 0; m < 3; m++) _rhs[0][m] = rtmp(m,i,j,k) - lhs(3,i,j,k)*_rhs[1][m] - lhs(4,i,j,k)*_rhs[2][m];
		//---------------------------------------------------------------------
		//      And the remaining two
		//---------------------------------------------------------------------
		_rhs[0][3] = rtmp(3,i,j,k) - lhsp(3,i,j,k)*_rhs[1][3] - lhsp(4,i,j,k)*_rhs[2][3];
		_rhs[0][4] = rtmp(4,i,j,k) - lhsm(3,i,j,k)*_rhs[1][4] - lhsm(4,i,j,k)*_rhs[2][4];

		if (k+2 < nz-1) {
			//---------------------------------------------------------------------
			//   block-diagonal matrix-vector multiplication (tzetar)
			//---------------------------------------------------------------------
			double xvel = us(i,j,k+2);
			double yvel = vs(i,j,k+2);
			double zvel = ws(i,j,k+2);
			double ac = speed(i,j,k+2);
			double uzik1 = u(0,i,j,k+2);
			double t1 = (bt*uzik1)/ac * (_rhs[2][3] + _rhs[2][4]);
			double t2 = _rhs[2][2] + t1;
			double t3 = bt*uzik1 * (_rhs[2][3] - _rhs[2][4]);

			_rhs[2][4] =  uzik1*(-xvel*_rhs[2][1] + yvel*_rhs[2][0]) + qs(i,j,k+2)*t2 + c2iv*(ac*ac)*t1 + zvel*t3;
			_rhs[2][3] =  zvel*t2  + t3;
			_rhs[2][2] =  uzik1*_rhs[2][0] + yvel*t2;
			_rhs[2][1] = -uzik1*_rhs[2][1] + xvel*t2;
			_rhs[2][0] = t2;
		}

		for (m = 0; m < 5; m++) {
			rhs(m,i,j,k+2) = _rhs[2][m];
			_rhs[2][m] = _rhs[1][m];
			_rhs[1][m] = _rhs[0][m];
		}
	}

	//---------------------------------------------------------------------
	//   block-diagonal matrix-vector multiplication (tzetar)
	//---------------------------------------------------------------------
	double xvel = us(i,j,1);
	double yvel = vs(i,j,1);
	double zvel = ws(i,j,1);
	double ac = speed(i,j,1);
	double uzik1 = u(0,i,j,1);
	double t1 = (bt*uzik1)/ac * (_rhs[2][3] + _rhs[2][4]);
	double t2 = _rhs[2][2] + t1;
	double t3 = bt*uzik1 * (_rhs[2][3] - _rhs[2][4]);

	rhs(4,i,j,1) =  uzik1*(-xvel*_rhs[2][1] + yvel*_rhs[2][0]) + qs(i,j,1)*t2 + c2iv*(ac*ac)*t1 + zvel*t3;
	rhs(3,i,j,1) =  zvel*t2  + t3;
	rhs(2,i,j,1) =  uzik1*_rhs[2][0] + yvel*t2;
	rhs(1,i,j,1) = -uzik1*_rhs[2][1] + xvel*t2;
	rhs(0,i,j,1) = t2;

	for (m = 0; m < 5; m++) rhs(m,i,j,0) = _rhs[1][m];
}
#undef lhs
#undef lhsp
#undef lhsm
#undef rtmp

void SP::z_solve () {
	int xblock = min(SOLVE_BLOCK,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int yblock = min(SOLVE_BLOCK/xblock,ny);
	int ygrid = (ny+yblock-1)/yblock;
	dim3 grid(ygrid,xgrid), block(yblock,xblock);

	START_TIMER(t_zsolve);
	z_solve_kernel<<<grid,block>>>(rho_i, us, vs, ws, speed, qs, u, rhs, lhs, rhstmp, nx, ny, nz);
	STOP_TIMER(t_zsolve);
}

//---------------------------------------------------------------------
// this function returns the exact solution at point xi, eta, zeta  
//---------------------------------------------------------------------
__device__ static void exact_solution_kernel (const double xi, const double eta, const double zeta, double *dtemp) {
	using namespace gpu_mod;
	for (int m = 0; m < 5; m++)
		dtemp[m] = ce[0][m] + xi*(ce[1][m] + xi*(ce[4][m] + xi*(ce[7][m] + xi*ce[10][m]))) +
				eta*(ce[2][m] + eta*(ce[5][m] + eta*(ce[8][m] + eta*ce[11][m])))+
				zeta*(ce[3][m] + zeta*(ce[6][m] + zeta*(ce[9][m] + zeta*ce[12][m])));
}

//---------------------------------------------------------------------
// compute the right hand side based on exact solution
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
	double xi, eta, zeta, dtemp[5], dtpp;
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
	double xi, eta, zeta, dtemp[5], dtpp;
	double ue[5][5], buf[3][5], cuf[3], q[3];

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
		for (m = 0; m < 5; m++) ue[j+1][m] = dtemp[m];
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
		dtemp[2] = forcing(2,i,j,k) - ty2*((ue[3][2]*buf[2][2]+c2*(ue[3][4]-q[2]))-(ue[1][2]*buf[0][2]+c2*(ue[1][4]-q[0])))+yycon1*(buf[2][2]-2.0*buf[1][2]+buf[0][2])+dy3ty1*( ue[3][2]-2.0*ue[2][2] +ue[1][2]);
		dtemp[3] = forcing(3,i,j,k) - ty2*(ue[3][3]*buf[2][2]-ue[1][3]*buf[0][2])+yycon2*(buf[2][3]-2.0*buf[1][3]+buf[0][3])+dy4ty1*( ue[3][3]-2.0*ue[2][3]+ ue[1][3]);
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

void SP::exact_rhs () {
	dim3 gridinit(ny,nz);
	exact_rhs_kernel_init<<<gridinit,nx>>>(forcing, nx, ny, nz);

	int yblock = min(ERHS_BLOCK,ny);
	int ygrid = (ny+yblock-1)/yblock;
	int zblock_y = min(ERHS_BLOCK/yblock,nz);
	int zgrid_y = (nz+zblock_y-1)/zblock_y;
	dim3 grid_x(zgrid_y,ygrid), block_x(zblock_y,yblock);
	exact_rhs_kernel_x<<<grid_x,block_x>>>(forcing, nx, ny, nz);

	int xblock = min(ERHS_BLOCK,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int zblock_x = min(ERHS_BLOCK/xblock,nz);
	int zgrid_x = (nz+zblock_x-1)/zblock_x;
	dim3 grid_y(zgrid_x,xgrid), block_y(zblock_x,xblock);
	exact_rhs_kernel_y<<<grid_y,block_y>>>(forcing, nx, ny, nz);

	int yblock_x = min(ERHS_BLOCK/xblock,ny);
	int ygrid_x = (ny+yblock_x-1)/yblock_x;
	dim3 grid_z(ygrid_x,xgrid), block_z(yblock_x,xblock);
	exact_rhs_kernel_z<<<grid_z,block_z>>>(forcing, nx, ny, nz);
}

//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
__global__ static void initialize_kernel (double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, temp[5];
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
	u(0,i,j,k) = 1.0;
	u(1,i,j,k) = 0.0;
	u(2,i,j,k) = 0.0;
	u(3,i,j,k) = 0.0;
	u(4,i,j,k) = 1.0;

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
	for (m = 0; m < 5; m++) {
		double Pxi = xi * Pface12[m] + (1.0-xi)*Pface11[m];
		double Peta = eta * Pface22[m] + (1.0-eta)*Pface21[m];
		double Pzeta = zeta * Pface32[m] + (1.0-zeta)*Pface31[m];
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
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// east face                                                      
	//---------------------------------------------------------------------
	xi = 1.0;
	if (i == nx-1) {
		zeta = (double)k * dnzm1;
		eta = (double)j * dnym1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// south face                                                 
	//---------------------------------------------------------------------
	eta = 0.0;
	if (j == 0) {
		zeta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zeta,temp);
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// north face                                    
	//---------------------------------------------------------------------
	eta = 1.0;
	if (j == ny-1) {
		zeta = (double)k * dnzm1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi,eta,zeta,temp);
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// bottom face                                       
	//---------------------------------------------------------------------
	zeta = 0.0;
	if (k == 0) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
	//---------------------------------------------------------------------
	// top face     
	//---------------------------------------------------------------------
	zeta = 1.0;
	if (k == nz-1) {
		eta = (double)j * dnym1;
		xi = (double)i * dnxm1;
		exact_solution_kernel (xi, eta, zeta, temp);
		for (m = 0; m < 5; m++) u(m,i,j,k) = temp[m];
	}
}

void SP::initialize () {
	dim3 grid(nz,ny);
	initialize_kernel<<<grid,nx>>> (u, nx, ny, nz);
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

void SP::error_norm () {
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
void SP::rhs_norm () {
	int xblock = min(64,nx);
	int xgrid = (nx+xblock-1)/xblock;
	int yblock = min(64/xblock,ny);
	int ygrid = (ny+yblock-1)/yblock;
	dim3 grid(ygrid,xgrid), block(yblock,xblock);

	rhs_norm_kernel<<<grid,block>>>(rmsbuf, rhs, nx, ny, nz);
	reduce_norm_kernel<<<1,NORM_BLOCK>>>(rmsbuf, nx, ny, nz);
	HANDLE_ERROR(cudaMemcpy(xcr, rmsbuf, 5*sizeof(double), cudaMemcpyDeviceToHost));
}

void SP::set_constants() {

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

	double bt = sqrt(0.5);

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
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::bt, &bt, sizeof(double)));
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
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttx1, &dttx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttx2, &dttx2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dtty1, &dtty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dtty2, &dtty2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttz1, &dttz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dttz2, &dttz2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dttx1, &c2dttx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dtty1, &c2dtty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::c2dttz1, &c2dttz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dt, &dt, sizeof(double)));
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

void SP::allocate_device_memory() {
	int gridsize = nx*ny*nz;
	int facesize = max(max(nx*ny, nx*nz), ny*nz);

	HANDLE_ERROR(cudaMalloc((void **)&u, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&forcing, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rhs, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rho_i, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&us, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&vs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&ws, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&qs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&speed, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&square, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&lhs, 9*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rhstmp, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rmsbuf, 5*facesize*sizeof(double)));
}

void SP::free_device_memory() {
	HANDLE_ERROR(cudaFree(u));
	HANDLE_ERROR(cudaFree(forcing));
	HANDLE_ERROR(cudaFree(rhs));
	HANDLE_ERROR(cudaFree(rho_i));
	HANDLE_ERROR(cudaFree(us));
	HANDLE_ERROR(cudaFree(vs));
	HANDLE_ERROR(cudaFree(ws));
	HANDLE_ERROR(cudaFree(qs));
	HANDLE_ERROR(cudaFree(speed));
	HANDLE_ERROR(cudaFree(square));
	HANDLE_ERROR(cudaFree(lhs));
	HANDLE_ERROR(cudaFree(rhstmp));
	HANDLE_ERROR(cudaFree(rmsbuf));
}

void SP::get_cuda_info() {
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
