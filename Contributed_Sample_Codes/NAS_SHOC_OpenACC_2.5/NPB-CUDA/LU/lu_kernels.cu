#include <stdio.h>
#include "main.h"

namespace gpu_mod {
// constants for LU method
__constant__ double dxi, deta, dzeta;
__constant__ double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3;
__constant__ double ce[13*5];
__constant__ double dt, omega;
}

// error handling
static void inline HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define START_TIMER(timer) if (timeron) { HANDLE_ERROR(cudaDeviceSynchronize()); timers->timer_start(timer); }
#define STOP_TIMER(timer) if (timeron) { HANDLE_ERROR(cudaDeviceSynchronize()); timers->timer_stop(timer); }

__global__ static void jacld_blts_kernel (const int plane, const int klower, const int jlower, const double *u, const double *rho_i, const double *qs, double *v, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double tmp1, tmp2, tmp3, tmat[5*5], tv[5];
	double r43, c1345, c34;

	k = klower+blockIdx.x+1;
	j = jlower+threadIdx.x+1;
	i = plane-k-j+3;
	if (j > ny-2 || i > nx-2 || i < 1) return;

	r43 = 4.0/3.0;
	c1345 = c1 * c3 * c4 * c5;
	c34 = c3 * c4;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//   form the first block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j,k-1);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*tz1*dz1;
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = -dt*tz2;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = -dt*tz2*(-(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2) - dt*tz1*(-c34*tmp2*u(1,i,j,k-1));
	tmat[1+5*1] = -dt*tz2*(u(3,i,j,k-1)*tmp1) - dt*tz1*c34*tmp1 - dt*tz1*dz2;
	tmat[1+5*2] = 0.0;
	tmat[1+5*3] = -dt*tz2*(u(1,i,j,k-1)*tmp1);
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = -dt*tz2*(-(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2) - dt*tz1*(-c34*tmp2*u(2,i,j,k-1));
	tmat[2+5*1] = 0.0;
	tmat[2+5*2] = -dt*tz2*(u(3,i,j,k-1)*tmp1) - dt*tz1*(c34*tmp1) - dt*tz1*dz3;
	tmat[2+5*3] = -dt*tz2*(u(2,i,j,k-1)*tmp1);
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = -dt*tz2*(-(u(3,i,j,k-1)*tmp1)*(u(3,i,j,k-1)*tmp1) + c2*qs(i,j,k-1)*tmp1) - dt*tz1*(-r43*c34*tmp2*u(3,i,j,k-1));
	tmat[3+5*1] = -dt*tz2*(-c2*(u(1,i,j,k-1)*tmp1));
	tmat[3+5*2] = -dt*tz2*(-c2*(u(2,i,j,k-1)*tmp1));
	tmat[3+5*3] = -dt*tz2*(2.0-c2)*(u(3,i,j,k-1)*tmp1) - dt*tz1*(r43*c34*tmp1) - dt*tz1*dz4;
	tmat[3+5*4] = -dt*tz2*c2;

	tmat[4+5*0] = -dt*tz2*((c2*2.0*qs(i,j,k-1)-c1*u(4,i,j,k-1))*u(3,i,j,k-1)*tmp2) - dt*tz1*(-(c34-c1345)*tmp3*(u(1,i,j,k-1)*u(1,i,j,k-1))-(c34-c1345)*tmp3*(u(2,i,j,k-1)*u(2,i,j,k-1))-(r43*c34-c1345)*tmp3*(u(3,i,j,k-1)*u(3,i,j,k-1))-c1345*tmp2*u(4,i,j,k-1));
	tmat[4+5*1] = -dt*tz2*(-c2*(u(1,i,j,k-1)*u(3,i,j,k-1))*tmp2) - dt*tz1*(c34-c1345)*tmp2*u(1,i,j,k-1);
	tmat[4+5*2] = -dt*tz2*(-c2*(u(2,i,j,k-1)*u(3,i,j,k-1))*tmp2) - dt*tz1*(c34-c1345)*tmp2*u(2,i,j,k-1);
	tmat[4+5*3] = -dt*tz2*(c1*(u(4,i,j,k-1)*tmp1)-c2*(qs(i,j,k-1)*tmp1+u(3,i,j,k-1)*u(3,i,j,k-1)*tmp2)) - dt*tz1*(r43*c34-c1345)*tmp2*u(3,i,j,k-1);
	tmat[4+5*4] = -dt*tz2*(c1*(u(3,i,j,k-1)*tmp1)) - dt*tz1*c1345*tmp1 - dt*tz1*dz5;

	for (m = 0; m < 5; m++) tv[m] = v(m,i,j,k) - omega*(tmat[m+5*0]*v(0,i,j,k-1) + tmat[m+5*1]*v(1,i,j,k-1) + tmat[m+5*2]*v(2,i,j,k-1) + tmat[m+5*3]*v(3,i,j,k-1) + tmat[m+5*4]*v(4,i,j,k-1));

	//---------------------------------------------------------------------
	//   form the second block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j-1,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*ty1*dy1;
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = -dt*ty2;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = -dt*ty2*(-(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2) - dt*ty1*(-c34*tmp2*u(1,i,j-1,k));
	tmat[1+5*1] = -dt*ty2*(u(2,i,j-1,k)*tmp1) - dt*ty1*(c34*tmp1) - dt*ty1*dy2;
	tmat[1+5*2] = -dt*ty2*(u(1,i,j-1,k)*tmp1);
	tmat[1+5*3] = 0.0;
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = -dt*ty2*(-(u(2,i,j-1,k)*tmp1)*(u(2,i,j-1,k)*tmp1) + c2*(qs(i,j-1,k)*tmp1)) - dt*ty1*(-r43*c34*tmp2*u(2,i,j-1,k));
	tmat[2+5*1] = -dt*ty2*(-c2*(u(1,i,j-1,k)*tmp1));
	tmat[2+5*2] = -dt*ty2*((2.0-c2)*(u(2,i,j-1,k)*tmp1)) - dt*ty1*(r43*c34*tmp1) - dt*ty1*dy3;
	tmat[2+5*3] = -dt*ty2*(-c2*(u(3,i,j-1,k)*tmp1));
	tmat[2+5*4] = -dt*ty2*c2;

	tmat[3+5*0] = -dt*ty2*(-(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2) - dt*ty1*(-c34*tmp2*u(3,i,j-1,k));
	tmat[3+5*1] = 0.0;
	tmat[3+5*2] = -dt*ty2*(u(3,i,j-1,k)*tmp1);
	tmat[3+5*3] = -dt*ty2*(u(2,i,j-1,k)*tmp1) - dt*ty1*(c34*tmp1) - dt*ty1*dy4;
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = -dt*ty2*((c2*2.0*qs(i,j-1,k)-c1*u(4,i,j-1,k))*(u(2,i,j-1,k)*tmp2)) - dt*ty1*(-(c34-c1345)*tmp3*(u(1,i,j-1,k)*u(1,i,j-1,k))-(r43*c34-c1345)*tmp3*(u(2,i,j-1,k)*u(2,i,j-1,k))-(c34-c1345)*tmp3*(u(3,i,j-1,k)*u(3,i,j-1,k))-c1345*tmp2*u(4,i,j-1,k));
	tmat[4+5*1] = -dt*ty2*(-c2*(u(1,i,j-1,k)*u(2,i,j-1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(1,i,j-1,k);
	tmat[4+5*2] = -dt*ty2*(c1*(u(4,i,j-1,k)*tmp1)-c2*(qs(i,j-1,k)*tmp1+u(2,i,j-1,k)*u(2,i,j-1,k)*tmp2)) - dt*ty1*(r43*c34-c1345)*tmp2*u(2,i,j-1,k);
	tmat[4+5*3] = -dt*ty2*(-c2*(u(2,i,j-1,k)*u(3,i,j-1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(3,i,j-1,k);
	tmat[4+5*4] = -dt*ty2*(c1*(u(2,i,j-1,k)*tmp1)) - dt*ty1*c1345*tmp1 - dt*ty1*dy5;

	for (m = 0; m < 5; m++) tv[m] = tv[m] - omega*(tmat[m+5*0]*v(0,i,j-1,k) + tmat[m+5*1]*v(1,i,j-1,k) + tmat[m+5*2]*v(2,i,j-1,k) + tmat[m+5*3]*v(3,i,j-1,k) + tmat[m+5*4]*v(4,i,j-1,k));

	//---------------------------------------------------------------------
	//   form the third block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i-1,j,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*tx1*dx1;
	tmat[0+5*1] = -dt*tx2;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = -dt*tx2*(-(u(1,i-1,j,k)*tmp1)*(u(1,i-1,j,k)*tmp1)+c2*qs(i-1,j,k)*tmp1) -dt*tx1*(-r43*c34*tmp2*u(1,i-1,j,k));
	tmat[1+5*1] = -dt*tx2*((2.0-c2)*(u(1,i-1,j,k)*tmp1)) - dt*tx1*(r43*c34*tmp1) - dt*tx1*dx2;
	tmat[1+5*2] = -dt*tx2*(-c2*(u(2,i-1,j,k)*tmp1));
	tmat[1+5*3] = -dt*tx2*(-c2*(u(3,i-1,j,k)*tmp1));
	tmat[1+5*4] = -dt*tx2*c2;

	tmat[2+5*0] = -dt*tx2*(-(u(1,i-1,j,k)*u(2,i-1,j,k))*tmp2) - dt*tx1*(-c34*tmp2*u(2,i-1,j,k));
	tmat[2+5*1] = -dt*tx2*(u(2,i-1,j,k)*tmp1);
	tmat[2+5*2] = -dt*tx2*(u(1,i-1,j,k)*tmp1) - dt*tx1*(c34*tmp1) - dt*tx1*dx3;
	tmat[2+5*3] = 0.0;
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = -dt*tx2*(-(u(1,i-1,j,k)*u(3,i-1,j,k))*tmp2) - dt*tx1*(-c34*tmp2*u(3,i-1,j,k));
	tmat[3+5*1] = -dt*tx2*(u(3,i-1,j,k)*tmp1);
	tmat[3+5*2] = 0.0;
	tmat[3+5*3] = -dt*tx2*(u(1,i-1,j,k)*tmp1) - dt*tx1*(c34*tmp1) - dt*tx1*dx4;
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = -dt*tx2*((c2*2.0*qs(i-1,j,k)-c1*u(4,i-1,j,k))*u(1,i-1,j,k)*tmp2) - dt*tx1*(-(r43*c34-c1345)*tmp3*(u(1,i-1,j,k)*u(1,i-1,j,k))-(c34-c1345)*tmp3*(u(2,i-1,j,k)*u(2,i-1,j,k))-(c34-c1345)*tmp3*(u(3,i-1,j,k)*u(3,i-1,j,k))-c1345*tmp2*u(4,i-1,j,k));
	tmat[4+5*1] = -dt*tx2*(c1*(u(4,i-1,j,k)*tmp1)-c2*(u(1,i-1,j,k)*u(1,i-1,j,k)*tmp2+qs(i-1,j,k)*tmp1)) - dt*tx1*(r43*c34-c1345)*tmp2*u(1,i-1,j,k);
	tmat[4+5*2] = -dt*tx2*(-c2*(u(2,i-1,j,k)*u(1,i-1,j,k))*tmp2) - dt*tx1*(c34-c1345)*tmp2*u(2,i-1,j,k);
	tmat[4+5*3] = -dt*tx2*(-c2*(u(3,i-1,j,k)*u(1,i-1,j,k))*tmp2) - dt*tx1*(c34-c1345)*tmp2*u(3,i-1,j,k);
	tmat[4+5*4] = -dt*tx2*(c1*(u(1,i-1,j,k)*tmp1)) - dt*tx1*c1345*tmp1 - dt*tx1*dx5;

	for (m = 0; m < 5; m++) tv[m] = tv[m] - omega*(tmat[m+0*5]*v(0,i-1,j,k) + tmat[m+5*1]*v(1,i-1,j,k) + tmat[m+5*2]*v(2,i-1,j,k) + tmat[m+5*3]*v(3,i-1,j,k) + tmat[m+5*4]*v(4,i-1,j,k));

	//---------------------------------------------------------------------
	//   form the block diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = 1.0 + dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1);
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = -dt*2.0*(tx1*r43+ty1+tz1)*c34*tmp2*u(1,i,j,k);
	tmat[1+5*1] = 1.0 + dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1) + dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2);
	tmat[1+5*2] = 0.0;
	tmat[1+5*3] = 0.0;
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = -dt*2.0*(tx1+ty1*r43+tz1)*c34*tmp2*u(2,i,j,k);
	tmat[2+5*1] = 0.0;
	tmat[2+5*2] = 1.0 + dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1) + dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3);
	tmat[2+5*3] = 0.0;
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = -dt*2.0*(tx1+ty1+tz1*r43)*c34*tmp2*u(3,i,j,k);
	tmat[3+5*1] = 0.0;
	tmat[3+5*2] = 0.0;
	tmat[3+5*3] = 1.0 + dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43) + dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4);
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = -dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u(1,i,j,k)*u(1,i,j,k))+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u(2,i,j,k)*u(2,i,j,k))+
			(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u(3,i,j,k)*u(3,i,j,k)))*tmp3+(tx1+ty1+tz1)*c1345*tmp2*u(4,i,j,k));
	tmat[4+5*1] = dt*2.0*tmp2*u(1,i,j,k)*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*2] = dt*2.0*tmp2*u(2,i,j,k)*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345));
	tmat[4+5*3] = dt*2.0*tmp2*u(3,i,j,k)*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345));
	tmat[4+5*4] = 1.0 + dt*2.0*(tx1+ty1+tz1)*c1345*tmp1 + dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5);

	//---------------------------------------------------------------------
	//   diagonal block inversion;  forward elimination
	//---------------------------------------------------------------------
	tmp1 = 1.0/tmat[0+0*5];
	tmp2 = tmp1*tmat[1+0*5];
	tmat[1+1*5] -= tmp2*tmat[0+1*5];
	tmat[1+2*5] -= tmp2*tmat[0+2*5];
	tmat[1+3*5] -= tmp2*tmat[0+3*5];
	tmat[1+4*5] -= tmp2*tmat[0+4*5];
	tv[1] -= tmp2*tv[0];

	tmp2 = tmp1*tmat[2+0*5];
	tmat[2+1*5] -= tmp2*tmat[0+1*5];
	tmat[2+2*5] -= tmp2*tmat[0+2*5];
	tmat[2+3*5] -= tmp2*tmat[0+3*5];
	tmat[2+4*5] -= tmp2*tmat[0+4*5];
	tv[2] -= tmp2*tv[0];

	tmp2 = tmp1*tmat[3+0*5];
	tmat[3+1*5] -= tmp2*tmat[0+1*5];
	tmat[3+2*5] -= tmp2*tmat[0+2*5];
	tmat[3+3*5] -= tmp2*tmat[0+3*5];
	tmat[3+4*5] -= tmp2*tmat[0+4*5];
	tv[3] -= tmp2*tv[0];

	tmp2 = tmp1*tmat[4+0*5];
	tmat[4+1*5] -= tmp2*tmat[0+1*5];
	tmat[4+2*5] -= tmp2*tmat[0+2*5];
	tmat[4+3*5] -= tmp2*tmat[0+3*5];
	tmat[4+4*5] -= tmp2*tmat[0+4*5];
	tv[4] -= tmp2*tv[0];

	tmp1 = 1.0/tmat[1+1*5];
	tmp2 = tmp1*tmat[2+1*5];
	tmat[2+2*5] -= tmp2*tmat[1+2*5];
	tmat[2+3*5] -= tmp2*tmat[1+3*5];
	tmat[2+4*5] -= tmp2*tmat[1+4*5];
	tv[2] -= tmp2*tv[1];

	tmp2 = tmp1*tmat[3+1*5];
	tmat[3+2*5] -= tmp2*tmat[1+2*5];
	tmat[3+3*5] -= tmp2*tmat[1+3*5];
	tmat[3+4*5] -= tmp2*tmat[1+4*5];
	tv[3] -= tmp2*tv[1];

	tmp2 = tmp1*tmat[4+1*5];
	tmat[4+2*5] -= tmp2*tmat[1+2*5];
	tmat[4+3*5] -= tmp2*tmat[1+3*5];
	tmat[4+4*5] -= tmp2*tmat[1+4*5];
	tv[4] -= tmp2*tv[1];

	tmp1 = 1.0/tmat[2+2*5];
	tmp2 = tmp1*tmat[3+2*5];
	tmat[3+3*5] -= tmp2*tmat[2+3*5];
	tmat[3+4*5] -= tmp2*tmat[2+4*5];
	tv[3] -= tmp2*tv[2];

	tmp2 = tmp1*tmat[4+2*5];
	tmat[4+3*5] -= tmp2*tmat[2+3*5];
	tmat[4+4*5] -= tmp2*tmat[2+4*5];
	tv[4] -= tmp2*tv[2];

	tmp1 = 1.0/tmat[3+3*5];
	tmp2 = tmp1*tmat[4+3*5];
	tmat[4+4*5] -= tmp2*tmat[3+4*5];
	tv[4] -= tmp2*tv[3];

	//---------------------------------------------------------------------
	//   back substitution
	//---------------------------------------------------------------------
	v(4,i,j,k) = tv[4]/tmat[4+4*5];

	tv[3] = tv[3] - tmat[3+4*5]*v(4,i,j,k);
	v(3,i,j,k) = tv[3]/tmat[3+3*5];

	tv[2] = tv[2] - tmat[2+3*5]*v(3,i,j,k) - tmat[2+4*5]*v(4,i,j,k);
	v(2,i,j,k) = tv[2]/tmat[2+2*5];

	tv[1] = tv[1] - tmat[1+2*5]*v(2,i,j,k) - tmat[1+3*5]*v(3,i,j,k) - tmat[1+4*5]*v(4,i,j,k);
	v(1,i,j,k) = tv[1]/tmat[1+1*5];

	tv[0] = tv[0] - tmat[0+1*5]*v(1,i,j,k) - tmat[0+2*5]*v(2,i,j,k) - tmat[0+3*5]*v(3,i,j,k) - tmat[0+4*5]*v(4,i,j,k);
	v(0,i,j,k) = tv[0]/tmat[0+0*5];
}

__global__ static void jacu_buts_kernel (const int plane, const int klower, const int jlower, const double *u, const double *rho_i, const double *qs, double *v, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double tmp, tmp1, tmp2, tmp3, tmat[5*5], tv[5];
	double r43, c1345, c34;

	k = klower+blockIdx.x+1;
	j = jlower+threadIdx.x+1;
	i = plane-j-k+3;
	if (i < 1 || i > nx-2 || j > ny-2) return;

	using namespace gpu_mod;

	r43 = 4.0/3.0;
	c1345 = c1*c3*c4*c5;
	c34 = c3*c4;

	//---------------------------------------------------------------------
	//   form the first block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i+1,j,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*tx1*dx1;
	tmat[0+5*1] = dt*tx2;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = dt*tx2*(-(u(1,i+1,j,k)*tmp1)*(u(1,i+1,j,k)*tmp1)+c2*qs(i+1,j,k)*tmp1) - dt*tx1*(-r43*c34*tmp2*u(1,i+1,j,k));
	tmat[1+5*1] = dt*tx2*((2.0-c2)*(u(1,i+1,j,k)*tmp1)) - dt*tx1*(r43*c34*tmp1) - dt*tx1*dx2;
	tmat[1+5*2] = dt*tx2*(-c2*(u(2,i+1,j,k)*tmp1));
	tmat[1+5*3] = dt*tx2*(-c2*(u(3,i+1,j,k)*tmp1));
	tmat[1+5*4] = dt*tx2*c2;

	tmat[2+5*0] = dt*tx2*(-(u(1,i+1,j,k)*u(2,i+1,j,k))*tmp2) - dt*tx1*(-c34*tmp2*u(2,i+1,j,k));
	tmat[2+5*1] = dt*tx2*(u(2,i+1,j,k)*tmp1);
	tmat[2+5*2] = dt*tx2*(u(1,i+1,j,k)*tmp1) - dt*tx1*(c34*tmp1) - dt*tx1*dx3;
	tmat[2+5*3] = 0.0;
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = dt*tx2*(-(u(1,i+1,j,k)*u(3,i+1,j,k))*tmp2) - dt*tx1*(-c34*tmp2*u(3,i+1,j,k));
	tmat[3+5*1] = dt*tx2*(u(3,i+1,j,k)*tmp1);
	tmat[3+5*2] = 0.0;
	tmat[3+5*3] = dt*tx2*(u(1,i+1,j,k)*tmp1) - dt*tx1*(c34*tmp1) - dt*tx1*dx4;
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = dt*tx2*((c2*2.0*qs(i+1,j,k)-c1*u(4,i+1,j,k))*(u(1,i+1,j,k)*tmp2)) - dt*tx1*(-(r43*c34-c1345)*tmp3*(u(1,i+1,j,k)*u(1,i+1,j,k))-(c34-c1345)*tmp3*(u(2,i+1,j,k)*u(2,i+1,j,k))-(c34-c1345)*tmp3*(u(3,i+1,j,k)*u(3,i+1,j,k))-c1345*tmp2*u(4,i+1,j,k));
	tmat[4+5*1] = dt*tx2*(c1*(u(4,i+1,j,k)*tmp1)-c2*(u(1,i+1,j,k)*u(1,i+1,j,k)*tmp2+qs(i+1,j,k)*tmp1)) - dt*tx1*(r43*c34-c1345)*tmp2*u(1,i+1,j,k);
	tmat[4+5*2] = dt*tx2*(-c2*(u(2,i+1,j,k)*u(1,i+1,j,k))*tmp2) - dt*tx1*(c34-c1345)*tmp2*u(2,i+1,j,k);
	tmat[4+5*3] = dt*tx2*(-c2*(u(3,i+1,j,k)*u(1,i+1,j,k))*tmp2) - dt*tx1*(c34-c1345)*tmp2*u(3,i+1,j,k);
	tmat[4+5*4] = dt*tx2*(c1*(u(1,i+1,j,k)*tmp1)) - dt*tx1*c1345*tmp1 - dt*tx1*dx5;

	for (m = 0; m < 5; m++) tv[m] = omega*(tmat[m+5*0]*v(0,i+1,j,k) + tmat[m+5*1]*v(1,i+1,j,k) + tmat[m+5*2]*v(2,i+1,j,k) + tmat[m+5*3]*v(3,i+1,j,k) + tmat[m+5*4]*v(4,i+1,j,k));

	//---------------------------------------------------------------------
	//   form the second block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j+1,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*ty1*dy1;
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = dt*ty2;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = dt*ty2*(-(u(1,i,j+1,k)*u(2,i,j+1,k))*tmp2) - dt*ty1*(-c34*tmp2*u(1,i,j+1,k));
	tmat[1+5*1] = dt*ty2*(u(2,i,j+1,k)*tmp1) - dt*ty1*(c34*tmp1) - dt*ty1*dy2;
	tmat[1+5*2] = dt*ty2*(u(1,i,j+1,k)*tmp1);
	tmat[1+5*3] = 0.0;
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = dt*ty2*(-(u(2,i,j+1,k)*tmp1)*(u(2,i,j+1,k)*tmp1)+c2*(qs(i,j+1,k)*tmp1)) - dt*ty1*(-r43*c34*tmp2*u(2,i,j+1,k));
	tmat[2+5*1] = dt*ty2*(-c2*(u(1,i,j+1,k)*tmp1));
	tmat[2+5*2] = dt*ty2*((2.0-c2)*(u(2,i,j+1,k)*tmp1)) - dt*ty1*(r43*c34*tmp1) - dt*ty1*dy3;
	tmat[2+5*3] = dt*ty2*(-c2*(u(3,i,j+1,k)*tmp1));
	tmat[2+5*4] = dt*ty2*c2;

	tmat[3+5*0] = dt*ty2*(-(u(2,i,j+1,k)*u(3,i,j+1,k))*tmp2) - dt*ty1*(-c34*tmp2*u(3,i,j+1,k));
	tmat[3+5*1] = 0.0;
	tmat[3+5*2] = dt*ty2*(u(3,i,j+1,k)*tmp1);
	tmat[3+5*3] = dt*ty2*(u(2,i,j+1,k)*tmp1) - dt*ty1*(c34*tmp1) - dt*ty1*dy4;
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = dt*ty2*((c2*2.0*qs(i,j+1,k)-c1*u(4,i,j+1,k))*(u(2,i,j+1,k)*tmp2)) - dt*ty1*(-(c34-c1345)*tmp3*(u(1,i,j+1,k)*u(1,i,j+1,k))-(r43*c34-c1345)*tmp3*(u(2,i,j+1,k)*u(2,i,j+1,k))-(c34-c1345)*tmp3*(u(3,i,j+1,k)*u(3,i,j+1,k))-c1345*tmp2*u(4,i,j+1,k));
	tmat[4+5*1] = dt*ty2*(-c2*(u(1,i,j+1,k)*u(2,i,j+1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(1,i,j+1,k);
	tmat[4+5*2] = dt*ty2*(c1*(u(4,i,j+1,k)*tmp1)-c2*(qs(i,j+1,k)*tmp1+u(2,i,j+1,k)*u(2,i,j+1,k)*tmp2)) - dt*ty1*(r43*c34-c1345)*tmp2*u(2,i,j+1,k);
	tmat[4+5*3] = dt*ty2*(-c2*(u(2,i,j+1,k)*u(3,i,j+1,k))*tmp2) - dt*ty1*(c34-c1345)*tmp2*u(3,i,j+1,k);
	tmat[4+5*4] = dt*ty2*(c1*(u(2,i,j+1,k)*tmp1)) - dt*ty1*c1345*tmp1 - dt*ty1*dy5;

	for (m = 0; m < 5; m++) tv[m]= tv[m] + omega*(tmat[m+5*0]*v(0,i,j+1,k) + tmat[m+5*1]*v(1,i,j+1,k) + tmat[m+5*2]*v(2,i,j+1,k) + tmat[m+5*3]*v(3,i,j+1,k) + tmat[m+5*4]*v(4,i,j+1,k));

	//---------------------------------------------------------------------
	//   form the third block sub-diagonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j,k+1);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = -dt*tz1*dz1;
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = dt*tz2;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = dt*tz2*(-(u(1,i,j,k+1)*u(3,i,j,k+1))*tmp2) - dt*tz1*(-c34*tmp2*u(1,i,j,k+1));
	tmat[1+5*1] = dt*tz2*(u(3,i,j,k+1)*tmp1) - dt*tz1*c34*tmp1 - dt*tz1*dz2;
	tmat[1+5*2] = 0.0;
	tmat[1+5*3] = dt*tz2*(u(1,i,j,k+1)*tmp1);
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = dt*tz2*(-(u(2,i,j,k+1)*u(3,i,j,k+1))*tmp2) - dt*tz1*(-c34*tmp2*u(2,i,j,k+1));
	tmat[2+5*1] = 0.0;
	tmat[2+5*2] = dt*tz2*(u(3,i,j,k+1)*tmp1) - dt*tz1*(c34*tmp1) - dt*tz1*dz3;
	tmat[2+5*3] = dt*tz2*(u(2,i,j,k+1)*tmp1);
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = dt*tz2*(-(u(3,i,j,k+1)*tmp1)*(u(3,i,j,k+1)*tmp1)+c2*(qs(i,j,k+1)*tmp1)) - dt*tz1*(-r43*c34*tmp2*u(3,i,j,k+1));
	tmat[3+5*1] = dt*tz2*(-c2*(u(1,i,j,k+1)*tmp1));
	tmat[3+5*2] = dt*tz2*(-c2*(u(2,i,j,k+1)*tmp1));
	tmat[3+5*3] = dt*tz2*(2.0-c2)*(u(3,i,j,k+1)*tmp1) - dt*tz1*(r43*c34*tmp1) - dt*tz1*dz4;
	tmat[3+5*4] = dt*tz2*c2;

	tmat[4+5*0] = dt*tz2*((c2*2.0*qs(i,j,k+1)-c1*u(4,i,j,k+1))*(u(3,i,j,k+1)*tmp2)) - dt*tz1*(-(c34-c1345)*tmp3*(u(1,i,j,k+1)*u(1,i,j,k+1))-(c34-c1345)*tmp3*(u(2,i,j,k+1)*u(2,i,j,k+1))-(r43*c34-c1345)*tmp3*(u(3,i,j,k+1)*u(3,i,j,k+1))-c1345*tmp2*u(4,i,j,k+1));
	tmat[4+5*1] = dt*tz2*(-c2*(u(1,i,j,k+1)*u(3,i,j,k+1))*tmp2) - dt*tz1*(c34-c1345)*tmp2*u(1,i,j,k+1);
	tmat[4+5*2] = dt*tz2*(-c2*(u(2,i,j,k+1)*u(3,i,j,k+1))*tmp2) - dt*tz1*(c34-c1345)*tmp2*u(2,i,j,k+1);
	tmat[4+5*3] = dt*tz2*(c1*(u(4,i,j,k+1)*tmp1)-c2*(qs(i,j,k+1)*tmp1+u(3,i,j,k+1)*u(3,i,j,k+1)*tmp2)) - dt*tz1*(r43*c34-c1345)*tmp2*u(3,i,j,k+1);
	tmat[4+5*4] = dt*tz2*(c1*(u(3,i,j,k+1)*tmp1)) - dt*tz1*c1345*tmp1 - dt*tz1*dz5;

	for (m = 0; m < 5; m++) tv[m] = tv[m] + omega*(tmat[m+5*0]*v(0,i,j,k+1) + tmat[m+5*1]*v(1,i,j,k+1) + tmat[m+5*2]*v(2,i,j,k+1) + tmat[m+5*3]*v(3,i,j,k+1) + tmat[m+5*4]*v(4,i,j,k+1));
		
	//---------------------------------------------------------------------
	//   form the block daigonal
	//---------------------------------------------------------------------
	tmp1 = rho_i(i,j,k);
	tmp2 = tmp1*tmp1;
	tmp3 = tmp1*tmp2;

	tmat[0+5*0] = 1.0 + dt*2.0*(tx1*dx1+ty1*dy1+tz1*dz1);
	tmat[0+5*1] = 0.0;
	tmat[0+5*2] = 0.0;
	tmat[0+5*3] = 0.0;
	tmat[0+5*4] = 0.0;

	tmat[1+5*0] = dt*2.0*(-tx1*r43-ty1-tz1)*(c34*tmp2*u(1,i,j,k));
	tmat[1+5*1] = 1.0 + dt*2.0*c34*tmp1*(tx1*r43+ty1+tz1) + dt*2.0*(tx1*dx2+ty1*dy2+tz1*dz2);
	tmat[1+5*2] = 0.0;
	tmat[1+5*3] = 0.0;
	tmat[1+5*4] = 0.0;

	tmat[2+5*0] = dt*2.0*(-tx1-ty1*r43-tz1)*(c34*tmp2*u(2,i,j,k));
	tmat[2+5*1] = 0.0;
	tmat[2+5*2] = 1.0 + dt*2.0*c34*tmp1*(tx1+ty1*r43+tz1) + dt*2.0*(tx1*dx3+ty1*dy3+tz1*dz3);
	tmat[2+5*3] = 0.0;
	tmat[2+5*4] = 0.0;

	tmat[3+5*0] = dt*2.0*(-tx1-ty1-tz1*r43)*(c34*tmp2*u(3,i,j,k));
	tmat[3+5*1] = 0.0;
	tmat[3+5*2] = 0.0;
	tmat[3+5*3] = 1.0 + dt*2.0*c34*tmp1*(tx1+ty1+tz1*r43) + dt*2.0*(tx1*dx4+ty1*dy4+tz1*dz4);
	tmat[3+5*4] = 0.0;

	tmat[4+5*0] = -dt*2.0*(((tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*(u(1,i,j,k)*u(1,i,j,k))+(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*(u(2,i,j,k)*u(2,i,j,k))+
			(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*(u(3,i,j,k)*u(3,i,j,k)))*tmp3 + (tx1+ty1+tz1)*c1345*tmp2*u(4,i,j,k));
	tmat[4+5*1] = dt*2.0*(tx1*(r43*c34-c1345)+ty1*(c34-c1345)+tz1*(c34-c1345))*tmp2*u(1,i,j,k);
	tmat[4+5*2] = dt*2.0*(tx1*(c34-c1345)+ty1*(r43*c34-c1345)+tz1*(c34-c1345))*tmp2*u(2,i,j,k);
	tmat[4+5*3] = dt*2.0*(tx1*(c34-c1345)+ty1*(c34-c1345)+tz1*(r43*c34-c1345))*tmp2*u(3,i,j,k);
	tmat[4+5*4] = 1.0 + dt*2.0*(tx1+ty1+tz1)*c1345*tmp1 + dt*2.0*(tx1*dx5+ty1*dy5+tz1*dz5);

	//---------------------------------------------------------------------
	//   diagonal block inversion
	//---------------------------------------------------------------------
	tmp1 = 1.0/tmat[0+0*5];
	tmp = tmp1*tmat[1+0*5];
	tmat[1+1*5] -= tmp*tmat[0+1*5];
	tmat[1+2*5] -= tmp*tmat[0+2*5];
	tmat[1+3*5] -= tmp*tmat[0+3*5];
	tmat[1+4*5] -= tmp*tmat[0+4*5];
	tv[1] -= tmp*tv[0];

	tmp = tmp1*tmat[2+0*5];
	tmat[2+1*5] -= tmp*tmat[0+1*5];
	tmat[2+2*5] -= tmp*tmat[0+2*5];
	tmat[2+3*5] -= tmp*tmat[0+3*5];
	tmat[2+4*5] -= tmp*tmat[0+4*5];
	tv[2] -= tmp*tv[0];

	tmp = tmp1*tmat[3+0*5];
	tmat[3+1*5] -= tmp*tmat[0+1*5];
	tmat[3+2*5] -= tmp*tmat[0+2*5];
	tmat[3+3*5] -= tmp*tmat[0+3*5];
	tmat[3+4*5] -= tmp*tmat[0+4*5];
	tv[3] -= tmp*tv[0];

	tmp = tmp1*tmat[4+0*5];
	tmat[4+1*5] -= tmp*tmat[0+1*5];
	tmat[4+2*5] -= tmp*tmat[0+2*5];
	tmat[4+3*5] -= tmp*tmat[0+3*5];
	tmat[4+4*5] -= tmp*tmat[0+4*5];
	tv[4] -= tmp*tv[0];

	tmp1 = 1.0/tmat[1+1*5];
	tmp = tmp1*tmat[2+1*5];
	tmat[2+2*5] -= tmp*tmat[1+2*5];
	tmat[2+3*5] -= tmp*tmat[1+3*5];
	tmat[2+4*5] -= tmp*tmat[1+4*5];
	tv[2] -= tmp*tv[1];

	tmp = tmp1*tmat[3+1*5];
	tmat[3+2*5] -= tmp*tmat[1+2*5];
	tmat[3+3*5] -= tmp*tmat[1+3*5];
	tmat[3+4*5] -= tmp*tmat[1+4*5];
	tv[3] -= tmp*tv[1];

	tmp = tmp1*tmat[4+1*5];
	tmat[4+2*5] -= tmp*tmat[1+2*5];
	tmat[4+3*5] -= tmp*tmat[1+3*5];
	tmat[4+4*5] -= tmp*tmat[1+4*5];
	tv[4] -= tmp*tv[1];

	tmp1 = 1.0/tmat[2+2*5];
	tmp = tmp1*tmat[3+2*5];
	tmat[3+3*5] -= tmp*tmat[2+3*5];
	tmat[3+4*5] -= tmp*tmat[2+4*5];
	tv[3] -= tmp*tv[2];

	tmp = tmp1*tmat[4+2*5];
	tmat[4+3*5] -= tmp*tmat[2+3*5];
	tmat[4+4*5] -= tmp*tmat[2+4*5];
	tv[4] -= tmp*tv[2];

	tmp1 = 1.0/tmat[3+3*5];
	tmp = tmp1 * tmat[4+3*5];
	tmat[4+4*5] -= tmp*tmat[3+4*5];
	tv[4] -= tmp*tv[3];

	//---------------------------------------------------------------------
	//   back substitution
	//---------------------------------------------------------------------
	tv[4] = tv[4]/tmat[4+4*5];

	tv[3] = tv[3] - tmat[3+4*5]*tv[4];
	tv[3] = tv[3]/tmat[3+3*5];

	tv[2] = tv[2] - tmat[2+3*5]*tv[3] - tmat[2+4*5]*tv[4];
	tv[2] = tv[2]/tmat[2+2*5];

	tv[1] = tv[1] - tmat[1+2*5]*tv[2] - tmat[1+3*5]*tv[3] - tmat[1+4*5]*tv[4];
	tv[1] = tv[1]/tmat[1+1*5];

	tv[0] = tv[0] - tmat[0+1*5]*tv[1] - tmat[0+2*5]*tv[2] - tmat[0+3*5]*tv[3] - tmat[0+4*5]*tv[4];
	tv[0] = tv[0]/tmat[0+0*5];

	v(0,i,j,k) -= tv[0];
	v(1,i,j,k) -= tv[1];
	v(2,i,j,k) -= tv[2];
	v(3,i,j,k) -= tv[3];
	v(4,i,j,k) -= tv[4];
}

__global__ static void ssor_kernel1 (double *rsd, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	
	i = threadIdx.x+1;
	j = blockIdx.y+1;
	k = blockIdx.x+1;
	m = threadIdx.y;
	using namespace gpu_mod;

	rsd(m,i,j,k) *= dt;
}

__global__ static void ssor_kernel2 (double *u, double *rsd, const double tmp, const int nx, const int ny, const int nz) {
	int i, j, k, m;

	i = threadIdx.x+1;
	j = blockIdx.y+1;
	k = blockIdx.x+1;

	for (m = 0; m < 5; m++) u(m,i,j,k) += tmp*rsd(m,i,j,k);
}

void LU::ssor(int niter) {
	dim3 grid_yz(nz-2,ny-2);
	dim3 grid_x(nx-2,5);
	double tmp = 1.0/(omega*(2.0-omega));

	//---------------------------------------------------------------------
	//   compute the steady-state residuals
	//---------------------------------------------------------------------
	rhs();

	//---------------------------------------------------------------------
	//   compute the L2 norms of newton iteration residuals
	//---------------------------------------------------------------------
	l2norm(rsd, rsdnm);

	for (int i = 0; i < t_last; i++) timers->timer_clear(i);
	HANDLE_ERROR(cudaDeviceSynchronize());
	timers->timer_start(0);

	//---------------------------------------------------------------------
	//   the timestep loop
	//---------------------------------------------------------------------
	for (int istep = 1; istep <= niter; istep++) {
		if ((istep % 20 == 0 || istep == itmax || istep == 1) && niter > 1)
			printf(" Time step %4d\n", istep);
		//---------------------------------------------------------------------
		//   perform SSOR iteration
		//---------------------------------------------------------------------
		START_TIMER(t_rhs);
		ssor_kernel1<<<grid_yz, grid_x>>>(rsd, nx, ny, nz);
		STOP_TIMER(t_rhs);

		//---------------------------------------------------------------------
		//   form the lower triangular part of the jacobian matrix
		//   perform the lower triangular solution
		//---------------------------------------------------------------------
		START_TIMER(t_jacld);
		for (int plane = 0; plane <= nx+ny+nz-9; plane++) {
			int klower = max(0, plane-(nx-3)-(ny-3));
			int kupper = min(plane, nz-3);
			int jlowermin = max(0, plane-kupper-(nx-3));
			int juppermax = min(plane, ny-3);

			jacld_blts_kernel<<<kupper-klower+1,juppermax-jlowermin+1>>>(plane, klower, jlowermin, u, rho_i, qs, rsd, nx, ny, nz);
		}
		STOP_TIMER(t_jacld);

		//---------------------------------------------------------------------
		//   form the strictly upper triangular part of the jacobian matrix
		//   perform the upper triangular solution
		//---------------------------------------------------------------------
		START_TIMER(t_jacu);
		for (int plane = nx+ny+nz-9; plane >= 0; plane--) {
			int klower = max(0, plane-(nx-3)-(ny-3));
			int kupper = min(plane, nz-3);
			int jlowermin = max(0, plane-kupper-(nx-3));
			int juppermax = min(plane, ny-3);

			jacu_buts_kernel<<<kupper-klower+1,juppermax-jlowermin+1>>>(plane, klower, jlowermin, u, rho_i, qs, rsd, nx, ny, nz);
		}
		STOP_TIMER(t_jacu);

		//---------------------------------------------------------------------
		//   update the variables
		//---------------------------------------------------------------------
		START_TIMER(t_add);
		ssor_kernel2<<<grid_yz,nx-2>>>(u, rsd, tmp, nx, ny, nz);
		STOP_TIMER(t_add);
	
		//---------------------------------------------------------------------
		//   compute the max-norms of newton iteration corrections
		//---------------------------------------------------------------------
		if (istep % inorm == 0) {
			double delunm[5];
			START_TIMER(t_l2norm);
			l2norm(rsd, delunm);
			STOP_TIMER(t_l2norm);
		}

		//---------------------------------------------------------------------
		//   compute the steady-state residuals
		//---------------------------------------------------------------------
		rhs();

		//---------------------------------------------------------------------
		//   compute the max-norms of newton iteration residuals
		//---------------------------------------------------------------------
		if (istep % inorm == 0) {
			START_TIMER(t_l2norm);
			l2norm(rsd, rsdnm);
			STOP_TIMER(t_l2norm);
		}

		//---------------------------------------------------------------------
		//   check the newton-iteration residuals against the tolerance levels
		//---------------------------------------------------------------------
		if (rsdnm[0] < tolrsd[0] && rsdnm[1] < tolrsd[1] && rsdnm[2] < tolrsd[2] && rsdnm[3] < tolrsd[3] && rsdnm[4] < tolrsd[4]) {
			printf("\n convergence was achieved after %4d pseudo-time steps\n", istep);
			break;
		}
	}

	HANDLE_ERROR(cudaDeviceSynchronize());
	timers->timer_stop(0);
	maxtime = timers->timer_read(0);
}

__global__ static void rhs_kernel_init (const double *u, double *rsd, const double *frct, double *qs, double *rho_i, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double tmp;

	k = blockIdx.x;
	j = blockIdx.y;
	i = threadIdx.x;

	for (m = 0; m < 5; m++) rsd(m,i,j,k) = -frct(m,i,j,k);
	rho_i(i,j,k) = tmp = 1.0/u(0,i,j,k);
	qs(i,j,k) = 0.5*(u(1,i,j,k)*u(1,i,j,k) + u(2,i,j,k)*u(2,i,j,k) + u(3,i,j,k)*u(3,i,j,k))*tmp;
}

__global__ static void rhs_kernel_x (const double *u, double *rsd, const double *qs, const double *rho_i, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u21;
	__shared__ double flux[RHSX_BLOCK][5];
	__shared__ double utmp[RHSX_BLOCK*5], rtmp[RHSX_BLOCK*5], rhotmp[RHSX_BLOCK];
	__shared__ double u21i[RHSX_BLOCK], u31i[RHSX_BLOCK], u41i[RHSX_BLOCK], u51i[RHSX_BLOCK];


	k = blockIdx.x+1;
	j = blockIdx.y+1;
	i = threadIdx.x;

	using namespace gpu_mod;

	while (i < nx) {
		// load u, rsd and rho_i using coalesced memory access 
		// first compute number of threads executing this region 
		nthreads = nx-(i-threadIdx.x);
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		utmp[m] = u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		utmp[m] = u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		utmp[m] = u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		utmp[m] = u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		utmp[m] = u(m%5, (i-threadIdx.x)+m/5, j, k);
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		rhotmp[threadIdx.x] = rho_i(i,j,k);
		__syncthreads();

		//---------------------------------------------------------------------
		//   xi-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = utmp[threadIdx.x*5+1];
		u21 = utmp[threadIdx.x*5+1]*rhotmp[threadIdx.x];
		q = qs(i,j,k);
		flux[threadIdx.x][1] = utmp[threadIdx.x*5+1]*u21 + c2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][2] = utmp[threadIdx.x*5+2]*u21;
		flux[threadIdx.x][3] = utmp[threadIdx.x*5+3]*u21;
		flux[threadIdx.x][4] = (c1*utmp[threadIdx.x*5+4]-c2*q)*u21;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSX_BLOCK-1 && i < nx-1) 
			for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] = rtmp[threadIdx.x*5+m] - tx2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);

		u21i[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31i[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41i[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51i[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();
	
		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = (4.0/3.0)*tx3*(u21i[threadIdx.x]-u21i[threadIdx.x-1]);
			flux[threadIdx.x][2] = tx3*(u31i[threadIdx.x]-u31i[threadIdx.x-1]);
			flux[threadIdx.x][3] = tx3*(u41i[threadIdx.x]-u41i[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*tx3*((u21i[threadIdx.x]*u21i[threadIdx.x]+u31i[threadIdx.x]*u31i[threadIdx.x]+u41i[threadIdx.x]*u41i[threadIdx.x]) - 
							(u21i[threadIdx.x-1]*u21i[threadIdx.x-1]+u31i[threadIdx.x-1]*u31i[threadIdx.x-1]+u41i[threadIdx.x-1]*u41i[threadIdx.x-1])) + 
							(1.0/6.0)*tx3*(u21i[threadIdx.x]*u21i[threadIdx.x]-u21i[threadIdx.x-1]*u21i[threadIdx.x-1]) + c1*c5*tx3*(u51i[threadIdx.x]-u51i[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSX_BLOCK-1 && i < nx-1) {
			rtmp[threadIdx.x*5+0] += dx1*tx1*(utmp[threadIdx.x*5-5]-2.0*utmp[threadIdx.x*5+0]+utmp[threadIdx.x*5+5]);
			rtmp[threadIdx.x*5+1] += tx3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dx2*tx1*(utmp[threadIdx.x*5-4]-2.0*utmp[threadIdx.x*5+1]+utmp[threadIdx.x*5+6]);
			rtmp[threadIdx.x*5+2] += tx3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dx3*tx1*(utmp[threadIdx.x*5-3]-2.0*utmp[threadIdx.x*5+2]+utmp[threadIdx.x*5+7]);
			rtmp[threadIdx.x*5+3] += tx3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dx4*tx1*(utmp[threadIdx.x*5-2]-2.0*utmp[threadIdx.x*5+3]+utmp[threadIdx.x*5+8]);
			rtmp[threadIdx.x*5+4] += tx3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dx5*tx1*(utmp[threadIdx.x*5-1]-2.0*utmp[threadIdx.x*5+4]+utmp[threadIdx.x*5+9]);

			//---------------------------------------------------------------------
			//   Fourth-order dissipation
			//---------------------------------------------------------------------
			if (i == 1) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,3,j,k));
			if (i == 2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,4,j,k));
			if (i >= 3 && i < nx-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i-2,j,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i+2,j,k));
			if (i == nx-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,nx-5,j,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);
			if (i == nx-2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,nx-4,j,k)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);
		}

		// store the updated rsd values using a coalesced write pattern
		// Note: this stores more values than actually computed but it leads to a more efficient execution
		m = threadIdx.x;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, (i-threadIdx.x)+m/5, j, k) = rtmp[m];

		i += RHSX_BLOCK-2;
	}
}

__global__ static void rhs_kernel_y (const double *u, double *rsd, const double *qs, const double *rho_i, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u31;
	__shared__ double flux[RHSY_BLOCK][5];
	__shared__ double utmp[RHSY_BLOCK*5], rtmp[RHSY_BLOCK*5], rhotmp[RHSY_BLOCK];
	__shared__ double u21j[RHSY_BLOCK], u31j[RHSY_BLOCK], u41j[RHSY_BLOCK], u51j[RHSY_BLOCK];

	k = blockIdx.x+1;
	i = blockIdx.y+1;
	j = threadIdx.x;

	using namespace gpu_mod;

	while(j < ny) {
		// load u, rsd and rho_i using coalesced memory access along the m-axis
		// first compute number of threads executing this region 
		nthreads = ny-(j-threadIdx.x);
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		utmp[m] = u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		utmp[m] = u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		utmp[m] = u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		utmp[m] = u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		utmp[m] = u(m%5, i, (j-threadIdx.x)+m/5, k);
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		rhotmp[threadIdx.x] = rho_i(i,j,k);
		__syncthreads();

		//---------------------------------------------------------------------
		//   eta-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = utmp[threadIdx.x*5+2];
		u31 = utmp[threadIdx.x*5+2] * rhotmp[threadIdx.x];
		q = qs(i,j,k);
		flux[threadIdx.x][1] = utmp[threadIdx.x*5+1]*u31;
		flux[threadIdx.x][2] = utmp[threadIdx.x*5+2]*u31 + c2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][3] = utmp[threadIdx.x*5+3]*u31;
		flux[threadIdx.x][4] = (c1*utmp[threadIdx.x*5+4]-c2*q)*u31;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSY_BLOCK-1 && j < ny-1) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] = rtmp[threadIdx.x*5+m] - ty2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);
		
		u21j[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31j[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41j[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51j[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();

		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = ty3*(u21j[threadIdx.x]-u21j[threadIdx.x-1]);
			flux[threadIdx.x][2] = (4.0/3.0)*ty3*(u31j[threadIdx.x]-u31j[threadIdx.x-1]);
			flux[threadIdx.x][3] = ty3*(u41j[threadIdx.x]-u41j[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*ty3*((u21j[threadIdx.x]*u21j[threadIdx.x]+u31j[threadIdx.x]*u31j[threadIdx.x]+u41j[threadIdx.x]*u41j[threadIdx.x]) - 
							(u21j[threadIdx.x-1]*u21j[threadIdx.x-1]+u31j[threadIdx.x-1]*u31j[threadIdx.x-1]+u41j[threadIdx.x-1]*u41j[threadIdx.x-1])) + 
							(1.0/6.0)*ty3*(u31j[threadIdx.x]*u31j[threadIdx.x]-u31j[threadIdx.x-1]*u31j[threadIdx.x-1]) + c1*c5*ty3*(u51j[threadIdx.x]-u51j[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSY_BLOCK-1 && j < ny-1) {
			rtmp[threadIdx.x*5+0] += dy1*ty1*(utmp[5*(threadIdx.x-1)]-2.0*utmp[threadIdx.x*5+0]+utmp[5*(threadIdx.x+1)]);
			rtmp[threadIdx.x*5+1] += ty3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dy2*ty1*(utmp[5*threadIdx.x-4]-2.0*utmp[threadIdx.x*5+1]+utmp[5*threadIdx.x+6]);
			rtmp[threadIdx.x*5+2] += ty3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dy3*ty1*(utmp[5*threadIdx.x-3]-2.0*utmp[threadIdx.x*5+2]+utmp[5*threadIdx.x+7]);
			rtmp[threadIdx.x*5+3] += ty3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dy4*ty1*(utmp[5*threadIdx.x-2]-2.0*utmp[threadIdx.x*5+3]+utmp[5*threadIdx.x+8]);
			rtmp[threadIdx.x*5+4] += ty3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dy5*ty1*(utmp[5*threadIdx.x-1]-2.0*utmp[threadIdx.x*5+4]+utmp[5*threadIdx.x+9]);

			//---------------------------------------------------------------------
			//   fourth-order dissipation
			//---------------------------------------------------------------------
			if (j == 1) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[5*threadIdx.x+m+5]+u(m,i,3,k));
			if (j == 2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,4,k));
			if (j >= 3 && j < ny-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,j-2,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j+2,k));
			if (j == ny-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,ny-5,k)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);
			if (j == ny-2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,ny-4,k)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);
		}

		// store the updated rsd values using a coalesced write pattern
		// Note: this stores more values than actually computed but it leads to a more efficient execution
		m = threadIdx.x;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, (j-threadIdx.x)+m/5, k) = rtmp[m];

		j += RHSY_BLOCK-2;
	}
}

__global__ static void rhs_kernel_z (const double *u, double *rsd, const double *qs, const double *rho_i, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u41;
	__shared__ double flux[RHSZ_BLOCK][5];
	__shared__ double utmp[RHSZ_BLOCK*5], rtmp[RHSZ_BLOCK*5], rhotmp[RHSZ_BLOCK];
	__shared__ double u21k[RHSZ_BLOCK], u31k[RHSZ_BLOCK], u41k[RHSZ_BLOCK], u51k[RHSZ_BLOCK];

	j = blockIdx.x+1;
	i = blockIdx.y+1;
	k = threadIdx.x;

	using namespace gpu_mod;

	while (k < nz) {
		// load u, rsd and rho_i using coalesced memory access along the m-axis
		// first compute number of threads executing this region 
		nthreads = (nz-(k-threadIdx.x));
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		utmp[m] = u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		utmp[m] = u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		utmp[m] = u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		utmp[m] = u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		utmp[m] = u(m%5, i, j, (k-threadIdx.x)+m/5);
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		rhotmp[threadIdx.x] = rho_i(i,j,k);
		__syncthreads();

		//---------------------------------------------------------------------
		//   zeta-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = utmp[threadIdx.x*5+3];
		u41 = utmp[threadIdx.x*5+3]*rhotmp[threadIdx.x];
		q = qs(i,j,k);
		flux[threadIdx.x][1] = utmp[threadIdx.x*5+1]*u41;
		flux[threadIdx.x][2] = utmp[threadIdx.x*5+2]*u41;
		flux[threadIdx.x][3] = utmp[threadIdx.x*5+3]*u41 + c2*(utmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][4] = (c1*utmp[threadIdx.x*5+4]-c2*q)*u41;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSZ_BLOCK-1 && k < nz-1) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] = rtmp[threadIdx.x*5+m] - tz2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);

		u21k[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+1];
		u31k[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+2];
		u41k[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+3];
		u51k[threadIdx.x] = rhotmp[threadIdx.x]*utmp[threadIdx.x*5+4];
		__syncthreads();

		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = tz3*(u21k[threadIdx.x]-u21k[threadIdx.x-1]);
			flux[threadIdx.x][2] = tz3*(u31k[threadIdx.x]-u31k[threadIdx.x-1]);
			flux[threadIdx.x][3] = (4.0/3.0)*tz3*(u41k[threadIdx.x]-u41k[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*tz3*((u21k[threadIdx.x]*u21k[threadIdx.x]+u31k[threadIdx.x]*u31k[threadIdx.x]+u41k[threadIdx.x]*u41k[threadIdx.x])-
							(u21k[threadIdx.x-1]*u21k[threadIdx.x-1]+u31k[threadIdx.x-1]*u31k[threadIdx.x-1]+u41k[threadIdx.x-1]*u41k[threadIdx.x-1])) + 
							(1.0/6.0)*tz3*(u41k[threadIdx.x]*u41k[threadIdx.x]-u41k[threadIdx.x-1]*u41k[threadIdx.x-1]) + c1*c5*tz3*(u51k[threadIdx.x]-u51k[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSZ_BLOCK-1 && k < nz-1) {
			rtmp[threadIdx.x*5+0] += dz1*tz1*(utmp[threadIdx.x*5-5]-2.0*utmp[threadIdx.x*5+0]+utmp[threadIdx.x*5+5]);
			rtmp[threadIdx.x*5+1] += tz3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dz2*tz1*(utmp[5*threadIdx.x-4]-2.0*utmp[threadIdx.x*5+1]+utmp[threadIdx.x*5+6]);
			rtmp[threadIdx.x*5+2] += tz3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dz3*tz1*(utmp[5*threadIdx.x-3]-2.0*utmp[threadIdx.x*5+2]+utmp[threadIdx.x*5+7]);
			rtmp[threadIdx.x*5+3] += tz3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dz4*tz1*(utmp[5*threadIdx.x-2]-2.0*utmp[threadIdx.x*5+3]+utmp[threadIdx.x*5+8]);
			rtmp[threadIdx.x*5+4] += tz3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dz5*tz1*(utmp[5*threadIdx.x-1]-2.0*utmp[threadIdx.x*5+4]+utmp[threadIdx.x*5+9]);

			//---------------------------------------------------------------------
			//   fourth-order dissipation
			//---------------------------------------------------------------------
			if (k == 1) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(5.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,3));
			if (k == 2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,4));
			if (k >= 3 && k < nz-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,j,k-2)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]+u(m,i,j,k+2));
			if (k == nz-3) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,j,nz-5)-4.0*utmp[threadIdx.x*5+m-5]+6.0*utmp[threadIdx.x*5+m]-4.0*utmp[threadIdx.x*5+m+5]);
			if (k == nz-2) for (m = 0; m < 5; m++) rtmp[threadIdx.x*5+m] -= dssp*(u(m,i,j,nz-4)-4.0*utmp[threadIdx.x*5+m-5]+5.0*utmp[threadIdx.x*5+m]);
		}

		// store the updated rsd values using a coalesced write pattern
		// Note: this stores more values than actually computed but it leads to a more efficient execution
		m = threadIdx.x;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5) = rtmp[m];
		m += nthreads;
		rsd(m%5, i, j, (k-threadIdx.x)+m/5) = rtmp[m];

		k += RHSZ_BLOCK-2;
	}
}

void LU::rhs() {
	dim3 grid(nz,ny);
	dim3 grid_yz(nz-2,ny-2);
	dim3 grid_xz(nz-2,nx-2);
	dim3 grid_xy(ny-2,nx-2);

	START_TIMER(t_rhs);
	rhs_kernel_init<<<grid,nx>>>(u, rsd, frct, qs, rho_i, nx, ny, nz);

	//---------------------------------------------------------------------
	//   xi-direction flux differences
	//---------------------------------------------------------------------
	START_TIMER(t_rhsx);
	rhs_kernel_x<<<grid_yz, min(nx,RHSX_BLOCK)>>>(u, rsd, qs, rho_i, nx, ny, nz);
	STOP_TIMER(t_rhsx);

	//---------------------------------------------------------------------
	//   eta-direction flux differences
	//---------------------------------------------------------------------
	START_TIMER(t_rhsy);
	rhs_kernel_y<<<grid_xz, min(ny,RHSY_BLOCK)>>>(u, rsd, qs, rho_i, nx, ny, nz);
	STOP_TIMER(t_rhsy);

	//---------------------------------------------------------------------
	//   zeta-direction flux differences
	//---------------------------------------------------------------------
	START_TIMER(t_rhsz);
	rhs_kernel_z<<<grid_xy, min(nz,RHSZ_BLOCK)>>>(u, rsd, qs, rho_i, nx, ny, nz);
	STOP_TIMER(t_rhsz);
	STOP_TIMER(t_rhs);
}

__global__ static void l2norm_kernel (const double *v, double *sum, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	__shared__ double sum_loc[5*NORM_BLOCK];

	k = blockIdx.x+1;
	j = blockIdx.y+1;
	i = threadIdx.x+1;
	for (m = 0; m < 5; m++) sum_loc[m+5*threadIdx.x] = 0.0;

	while (i < nx-1) {
		for (m = 0; m < 5; m++)	sum_loc[m+5*threadIdx.x] += v(m,i,j,k)*v(m,i,j,k);
		i += blockDim.x;
	}

	// reduction in x direction
	i = threadIdx.x;
	int loc_max = blockDim.x;
	int dist = (loc_max+1)/2;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max)
			for (m = 0; m < 5; m++) sum_loc[m+5*i] += sum_loc[m+5*(i+dist)];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}

	if (i == 0) for (m = 0; m < 5; m++) sum[m+5*(blockIdx.y+gridDim.y*blockIdx.x)] = sum_loc[m];
}

__global__ static void norm_reduce(double *rms, const int size) {
	int i, m, loc_max, dist;
	__shared__ double buffer[5*NORM_BLOCK];

	i = threadIdx.x;
	for (m = 0; m < 5; m++) buffer[m+5*i] = 0.0;

	while (i < size) {
		for (m = 0; m < 5; m++) buffer[m+5*threadIdx.x] += rms[m+5*i];
		i += blockDim.x;
	}

	loc_max = blockDim.x;
	dist = (loc_max+1)/2;
	i = threadIdx.x;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max) 
			for (m = 0; m < 5; m++) buffer[m+5*i] += buffer[m+5*(i+dist)];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}

	if (threadIdx.x < 5) rms[threadIdx.x] = buffer[threadIdx.x];
}

void LU::l2norm (const double *v, double *sum) {
	dim3 grid(nz-2,ny-2);

	l2norm_kernel<<<grid,min(nx-2,NORM_BLOCK)>>>(v, dev_norm_buf, nx, ny, nz);
	norm_reduce<<<1,NORM_BLOCK>>>(dev_norm_buf, (nz-2)*(ny-2));
	HANDLE_ERROR(cudaMemcpy(sum, dev_norm_buf, 5*sizeof(double), cudaMemcpyDeviceToHost));

	for (int m = 0; m < 5; m++) sum[m] = sqrt(sum[m]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));
}

//---------------------------------------------------------------------
//
//   compute the exact solution at (i,j,k)
//
//---------------------------------------------------------------------
__device__ static void exact_kernel (const int i, const int j, const int k, double *u000ijk, const int nx, const int ny, const int nz) {
	int m;
	double xi, eta, zeta;

	using namespace gpu_mod;

	xi = (double)i/(double)(nx-1);
	eta = (double)j/(double)(ny-1);
	zeta = (double)k/(double)(nz-1);

	for (m = 0; m < 5; m++) u000ijk[m] = ce[m+0*5]+(ce[m+1*5]+(ce[m+4*5]+(ce[m+7*5]+ce[m+10*5]*xi)*xi)*xi)*xi + (ce[m+2*5]+(ce[m+5*5]+(ce[m+8*5]+ce[m+11*5]*eta)*eta)*eta)*eta + (ce[m+3*5]+(ce[m+6*5]+(ce[m+9*5]+ce[m+12*5]*zeta)*zeta)*zeta)*zeta;
}

//---------------------------------------------------------------------
//
//   compute the solution error
//
//---------------------------------------------------------------------
__global__ static void error_kernel (const double *u, double *errnm, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double tmp, u000ijk[5];
	__shared__ double errnm_loc[5*NORM_BLOCK];

	k = blockIdx.x+1;
	j = blockIdx.y+1;
	i = threadIdx.x+1;
	for (m = 0; m < 5; m++) errnm_loc[m+5*threadIdx.x] = 0.0;

	while (i < nx-1) {
		exact_kernel(i, j, k, u000ijk, nx, ny, nz);
		for (m = 0; m < 5; m++) {
			tmp = u000ijk[m]-u(m,i,j,k);
			errnm_loc[m+5*threadIdx.x] += tmp * tmp;
		}
		i += blockDim.x;
	}

	// reduce in x direction
	i = threadIdx.x;
	int loc_max = blockDim.x;
	int dist = (loc_max+1)/2;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max)
			for (m = 0; m < 5; m++) errnm_loc[m+5*i] += errnm_loc[m+5*(i+dist)];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}

	if (i == 0) for (m = 0; m < 5; m++) errnm[m+5*(blockIdx.y+gridDim.y*blockIdx.x)] = errnm_loc[m];
}

void LU::error() {
	dim3 grid(nz-2,ny-2);

	error_kernel<<<grid, min(nx-2,NORM_BLOCK)>>>(u, dev_norm_buf, nx, ny, nz);
	norm_reduce<<<1,NORM_BLOCK>>>(dev_norm_buf, (nz-2)*(ny-2));
	HANDLE_ERROR(cudaMemcpy(errnm, dev_norm_buf, 5*sizeof(double), cudaMemcpyDeviceToHost));

	for (int m = 0; m < 5; m++) errnm[m] = sqrt(errnm[m]/((double)(nz-2)*(double)(ny-2)*(double)(nx-2)));
}

__global__ static void pintgr_kernel_1 (const double *u, double *frc, const int nx, const int ny, const int nz) {
	int i, j, k;
	__shared__ double phi1[PINTGR_BLOCK][PINTGR_BLOCK], phi2[PINTGR_BLOCK][PINTGR_BLOCK], frc1[PINTGR_BLOCK*PINTGR_BLOCK];

	i = blockIdx.x*(PINTGR_BLOCK-1)+threadIdx.x+1;
	j = blockIdx.y*(PINTGR_BLOCK-1)+threadIdx.y+1;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//   initialize
	//---------------------------------------------------------------------
	if (j < ny-2 && i < nx-1) {
		k = 2;
		phi1[threadIdx.x][threadIdx.y] = c2*(u(4,i,j,k) - 0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))/u(0,i,j,k));
		k = nz-2;
		phi2[threadIdx.x][threadIdx.y] = c2*(u(4,i,j,k) - 0.5*(u(1,i,j,k)*u(1,i,j,k)+u(2,i,j,k)*u(2,i,j,k)+u(3,i,j,k)*u(3,i,j,k))/u(0,i,j,k));
	}
	__syncthreads();

	frc1[threadIdx.y*blockDim.x+threadIdx.x] = 0.0;
	if (j < ny-3 && i < nx-2 && threadIdx.x < PINTGR_BLOCK-1 && threadIdx.y < PINTGR_BLOCK-1) 
		frc1[threadIdx.y*blockDim.x+threadIdx.x] = phi1[threadIdx.x][threadIdx.y]+phi1[threadIdx.x+1][threadIdx.y]+phi1[threadIdx.x][threadIdx.y+1]+phi1[threadIdx.x+1][threadIdx.y+1]+
								phi2[threadIdx.x][threadIdx.y]+phi2[threadIdx.x+1][threadIdx.y]+phi2[threadIdx.x][threadIdx.y+1]+phi2[threadIdx.x+1][threadIdx.y+1];

	// reduce
	int loc_max = blockDim.x*blockDim.y;
	int dist = (loc_max+1)/2;
	i = threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max) frc1[i] += frc1[i+dist];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}
	if (i == 0) frc[blockIdx.y*gridDim.x+blockIdx.x] = frc1[0]*dxi*deta;
}

__global__ static void pintgr_kernel_2 (const double *u, double *frc, const int nx, const int ny, const int nz) {
	int i, j, k, kp, ip;
	__shared__ double phi1[PINTGR_BLOCK][PINTGR_BLOCK], phi2[PINTGR_BLOCK][PINTGR_BLOCK], frc2[PINTGR_BLOCK*PINTGR_BLOCK];

	i = blockIdx.x*(PINTGR_BLOCK-1)+1;
	k = blockIdx.y*(PINTGR_BLOCK-1)+2;
	kp = threadIdx.y;
	ip = threadIdx.x;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//   initialize
	//---------------------------------------------------------------------
	if (k+kp < nz-1 && i+ip < nx-1) {
		j = 1;
		phi1[kp][ip] = c2*(u(4,i+ip,j,k+kp) - 0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
		j = ny-3;
		phi2[kp][ip] = c2*(u(4,i+ip,j,k+kp) - 0.5*(u(1,i+ip,j,k+kp)*u(1,i+ip,j,k+kp)+u(2,i+ip,j,k+kp)*u(2,i+ip,j,k+kp)+u(3,i+ip,j,k+kp)*u(3,i+ip,j,k+kp))/u(0,i+ip,j,k+kp));
	}
	__syncthreads();

	frc2[kp*PINTGR_BLOCK+ip] = 0.0;
	if (k+kp < nz-2 && i+ip < nx-2 && kp < PINTGR_BLOCK-1 && ip < PINTGR_BLOCK-1)
		frc2[kp*PINTGR_BLOCK+ip] += phi1[kp][ip] + phi1[kp+1][ip] + phi1[kp][ip+1] + phi1[kp+1][ip+1] + phi2[kp][ip] + phi2[kp+1][ip] + phi2[kp][ip+1] + phi2[kp+1][ip+1];

	// reduce
	int loc_max = blockDim.x*blockDim.y;
	int dist = (loc_max+1)/2;
	i = threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max) frc2[i] += frc2[i+dist];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}
	if (i == 0) frc[blockIdx.y*gridDim.x+blockIdx.x] = frc2[0]*dxi*dzeta;
}

__global__ static void pintgr_kernel_3 (const double *u, double *frc, const int nx, const int ny, const int nz) {
	int j, k, jp, kp;
	__shared__ double phi1[PINTGR_BLOCK][PINTGR_BLOCK], phi2[PINTGR_BLOCK][PINTGR_BLOCK], frc3[PINTGR_BLOCK*PINTGR_BLOCK];

	j = blockIdx.x*(PINTGR_BLOCK-1)+1;
	k = blockIdx.y*(PINTGR_BLOCK-1)+2;
	kp = threadIdx.y;
	jp = threadIdx.x;

	using namespace gpu_mod;

	//---------------------------------------------------------------------
	//   initialize
	//---------------------------------------------------------------------
	if (k+kp < nz-1 && j+jp < ny-2) {
		phi1[kp][jp] = c2*(u(4,1,j+jp,k+kp) - 0.5*(u(1,1,j+jp,k+kp)*u(1,1,j+jp,k+kp)+u(2,1,j+jp,k+kp)*u(2,1,j+jp,k+kp)+u(3,1,j+jp,k+kp)*u(3,1,j+jp,k+kp))/u(0,1,j+jp,k+kp));
		phi2[kp][jp] = c2*(u(4,nx-2,j+jp,k+kp) - 0.5*(u(1,nx-2,j+jp,k+kp)*u(1,nx-2,j+jp,k+kp)+u(2,nx-2,j+jp,k+kp)*u(2,nx-2,j+jp,k+kp)+u(3,nx-2,j+jp,k+kp)*u(3,nx-2,j+jp,k+kp))/u(0,nx-2,j+jp,k+kp));
	}
	__syncthreads();

	frc3[kp*PINTGR_BLOCK+jp] = 0.0;
	if (k+kp < nz-2 && j+jp < ny-3 && kp < PINTGR_BLOCK-1 && jp < PINTGR_BLOCK-1)
		frc3[kp*PINTGR_BLOCK+jp] = phi1[kp][jp] + phi1[kp+1][jp] + phi1[kp][jp+1] + phi1[kp+1][jp+1] + phi2[kp][jp] + phi2[kp+1][jp] + phi2[kp][jp+1] + phi2[kp+1][jp+1];

	// reduce
	int loc_max = blockDim.x*blockDim.y;
	int dist = (loc_max+1)/2;
	j = threadIdx.y*blockDim.x+threadIdx.x;
	__syncthreads();
	while (loc_max > 1) {
		if (j < dist && j+dist < loc_max) frc3[j] += frc3[j+dist];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}
	if (j == 0) frc[blockIdx.y*gridDim.x+blockIdx.x] = frc3[0]*deta*dzeta;
}

__global__ static void pintgr_reduce (double *frc, const int num) {
	int i, loc_max, dist;
	__shared__ double buffer[PINTGR_BLOCK*PINTGR_BLOCK];

	i = threadIdx.x;
	buffer[i] = 0.0;

	while (i < num) {
		buffer[threadIdx.x] += frc[i];
		i += blockDim.x;
	}

	loc_max = blockDim.x;
	dist = (loc_max+1)/2;
	i = threadIdx.x;
	__syncthreads();
	while (loc_max > 1) {
		if (i < dist && i+dist < loc_max) buffer[i] += buffer[i+dist];
		loc_max = dist;
		dist = (dist+1)/2;
		__syncthreads();
	}

	if (i == 0) frc[0] = .25*buffer[0];
}

void LU::pintgr() {
	dim3 grid(PINTGR_BLOCK,PINTGR_BLOCK);
	dim3 grid_xy((nx-3+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1), (ny-4+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1));
	int grid1_size = grid_xy.x*grid_xy.y;
	pintgr_kernel_1<<<grid_xy,grid>>>(u, dev_norm_buf, nx, ny, nz);

	dim3 grid_xz((nx-3+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1), (nz-4+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1));
	int grid2_size = grid_xz.x*grid_xz.y;
	pintgr_kernel_2<<<grid_xz,grid>>>(u, dev_norm_buf+grid1_size, nx, ny, nz);

	dim3 grid_yz((ny-4+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1), (nz-4+PINTGR_BLOCK-2)/(PINTGR_BLOCK-1));
	int grid3_size = grid_yz.x*grid_yz.y;
	pintgr_kernel_3<<<grid_yz,grid>>>(u, dev_norm_buf+grid1_size+grid2_size, nx, ny, nz);

	pintgr_reduce<<<1,PINTGR_BLOCK*PINTGR_BLOCK>>>(dev_norm_buf, grid1_size+grid2_size+grid3_size);
	HANDLE_ERROR(cudaMemcpy(&frc, dev_norm_buf, sizeof(double), cudaMemcpyDeviceToHost));
}

__global__ static void setbv_kernel_x (double *u, const int nx, const int ny, const int nz) {
	int j, k, m;
	double temp1[5], temp2[5];

	k = blockIdx.x;
	j = threadIdx.x;

	//---------------------------------------------------------------------
	//   set the dependent variable values along east and west faces
	//---------------------------------------------------------------------
	exact_kernel(0, j, k, temp1, nx, ny, nz);
	exact_kernel(nx-1, j, k, temp2, nx, ny, nz);
	for (m = 0; m < 5; m++) {
		u(m,0,j,k) = temp1[m];
		u(m,nx-1,j,k) = temp2[m];
	}
}

__global__ static void setbv_kernel_y (double *u, const int nx, const int ny, const int nz) {
	int i, k, m;
	double temp1[5], temp2[5];

	k = blockIdx.x;
	i = threadIdx.x;

	//---------------------------------------------------------------------
	//   set the dependent variable values along north and south faces
	//---------------------------------------------------------------------
	exact_kernel(i, 0, k, temp1, nx, ny, nz);
	exact_kernel(i, ny-1, k, temp2, nx, ny, nz);
	for (m = 0; m < 5; m++) {
		u(m,i,0,k) = temp1[m];
		u(m,i,ny-1,k) = temp2[m];
	}
}

__global__ static void setbv_kernel_z (double *u, const int nx, const int ny, const int nz) {
	int i, j, m;
	double temp1[5], temp2[5];

	j = blockIdx.x;
	i = threadIdx.x;

	//---------------------------------------------------------------------
	//   set the dependent variable values along the top and bottom faces
	//---------------------------------------------------------------------
	exact_kernel(i, j, 0, temp1, nx, ny, nz);
	exact_kernel(i, j, nz-1, temp2, nx, ny, nz);
	for (m = 0; m < 5; m++) {
		u(m,i,j,0) = temp1[m];
		u(m,i,j,nz-1) = temp2[m];
	}
}

//---------------------------------------------------------------------
//   set the boundary values of dependent variables
//---------------------------------------------------------------------
void LU::setbv() {
	setbv_kernel_z<<<ny,nx>>>(u, nx, ny, nz);
	setbv_kernel_y<<<nz,nx>>>(u, nx, ny, nz);
	setbv_kernel_x<<<nz,ny>>>(u, nx, ny, nz);
}

__global__ static void setiv_kernel (double *u, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta, pxi, peta, pzeta;
	double ue_1jk[5], ue_nx0jk[5], ue_i1k[5], ue_iny0k[5], ue_ij1[5], ue_ijnz[5];

	k = blockIdx.x+1;
	j = blockIdx.y+1;
	i = threadIdx.x+1;

	zeta = (double)k/(double)(nz-1);
	eta = (double)j/(double)(ny-1);
	xi = (double)i/(double)(nx-1);
	exact_kernel(0, j, k, ue_1jk, nx, ny, nz);
	exact_kernel(nx-1, j, k, ue_nx0jk, nx, ny, nz);
	exact_kernel(i, 0, k, ue_i1k, nx, ny, nz);
	exact_kernel(i, ny-1, k, ue_iny0k, nx, ny, nz);
	exact_kernel(i, j, 0, ue_ij1, nx, ny, nz);
	exact_kernel(i, j, nz-1, ue_ijnz, nx, ny, nz);
	for (m = 0; m < 5; m++) {
		pxi = (1.0-xi)*ue_1jk[m] + xi*ue_nx0jk[m];
		peta = (1.0-eta)*ue_i1k[m] + eta*ue_iny0k[m];
		pzeta = (1.0-zeta)*ue_ij1[m] + zeta*ue_ijnz[m];

		u(m,i,j,k) = pxi + peta + pzeta - pxi*peta - peta*pzeta - pzeta*pxi + pxi*peta*pzeta;
	}
}

//---------------------------------------------------------------------
//
//   set the initial values of independent variables based on tri-linear
//   interpolation of boundary values in the computational space.
//
//---------------------------------------------------------------------
void LU::setiv() {
	dim3 grid(nz-2, ny-2);
	setiv_kernel<<<grid,nx-2>>>(u, nx, ny, nz);
}

__global__ static void erhs_kernel_init (double *frct, double *rsd, const int nx, const int ny, const int nz) {
	int i, j, k, m;
	double xi, eta, zeta;

	k = blockIdx.x;
	j = blockIdx.y;
	i = threadIdx.x;

	using namespace gpu_mod;

	for (m = 0; m < 5; m++) frct(m,i,j,k) = 0.0;

	zeta = (double)k/((double)(nz-1));
	eta = (double)j/((double)(ny-1));
	xi = (double)i/((double)(nx-1));
	for (m = 0; m < 5; m++) rsd(m,i,j,k) = ce[m+0*5] + (ce[m+1*5]+(ce[m+4*5]+(ce[m+7*5]+ce[m+10*5]*xi)*xi)*xi)*xi + (ce[m+2*5]+(ce[m+5*5]+(ce[m+8*5]+ce[m+11*5]*eta)*eta)*eta)*eta + (ce[m+3*5]+(ce[m+6*5]+(ce[m+9*5]+ce[m+12*5]*zeta)*zeta)*zeta)*zeta;
}

__global__ static void erhs_kernel_x (double *frct, const double *rsd, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u21;
	__shared__ double flux[RHSX_BLOCK][5];
	__shared__ double rtmp[RHSX_BLOCK*5];
	__shared__ double u21i[RHSX_BLOCK], u31i[RHSX_BLOCK], u41i[RHSX_BLOCK], u51i[RHSX_BLOCK];
	double utmp[5];

	k = blockIdx.x+1;
	j = blockIdx.y+1;
	i = threadIdx.x;

	using namespace gpu_mod;

	while (i < nx) {
		// load rsd using coalesced memory access 
		// first compute number of threads executing this region 
		nthreads = nx-(i-threadIdx.x);
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, (i-threadIdx.x)+m/5, j, k);
		__syncthreads();

		//---------------------------------------------------------------------
		//   xi-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = rtmp[threadIdx.x*5+1];
		u21 = rtmp[threadIdx.x*5+1]/rtmp[threadIdx.x*5+0];
		q = 0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1] + rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2] + rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x][1] = rtmp[threadIdx.x*5+1]*u21 + c2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][2] = rtmp[threadIdx.x*5+2]*u21;
		flux[threadIdx.x][3] = rtmp[threadIdx.x*5+3]*u21;
		flux[threadIdx.x][4] = (c1*rtmp[threadIdx.x*5+4] - c2*q)*u21;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSX_BLOCK-1 && i < nx-1)
			for (m = 0; m < 5; m++) utmp[m] = frct(m,i,j,k) - tx2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);

		u21 = 1.0/rtmp[threadIdx.x*5+0];
		u21i[threadIdx.x] = u21*rtmp[threadIdx.x*5+1];
		u31i[threadIdx.x] = u21*rtmp[threadIdx.x*5+2];
		u41i[threadIdx.x] = u21*rtmp[threadIdx.x*5+3];
		u51i[threadIdx.x] = u21*rtmp[threadIdx.x*5+4];
		__syncthreads();

		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = (4.0/3.0)*tx3*(u21i[threadIdx.x]-u21i[threadIdx.x-1]);
			flux[threadIdx.x][2] = tx3*(u31i[threadIdx.x]-u31i[threadIdx.x-1]);
			flux[threadIdx.x][3] = tx3*(u41i[threadIdx.x]-u41i[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*tx3*((u21i[threadIdx.x]*u21i[threadIdx.x]+u31i[threadIdx.x]*u31i[threadIdx.x]+u41i[threadIdx.x]*u41i[threadIdx.x]) - 
							(u21i[threadIdx.x-1]*u21i[threadIdx.x-1]+u31i[threadIdx.x-1]*u31i[threadIdx.x-1]+u41i[threadIdx.x-1]*u41i[threadIdx.x-1])) + 
							(1.0/6.0)*tx3*(u21i[threadIdx.x]*u21i[threadIdx.x]-u21i[threadIdx.x-1]*u21i[threadIdx.x-1]) + c1*c5*tx3*(u51i[threadIdx.x]-u51i[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSX_BLOCK-1 && i < nx-1) {
			utmp[0] += dx1*tx1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1] += tx3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dx2*tx1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2] += tx3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dx3*tx1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3] += tx3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dx4*tx1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4] += tx3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dx5*tx1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);
		
			//---------------------------------------------------------------------
			//   Fourth-order dissipation
			//---------------------------------------------------------------------
			if (i == 1) for (m = 0; m < 5; m++) frct(m,1,j,k) = utmp[m] - dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,3,j,k));
			if (i == 2) for (m = 0; m < 5; m++) frct(m,2,j,k) = utmp[m] - dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,4,j,k));
			if (i >= 3 && i < nx-3) for (m = 0; m < 5; m++) frct(m,i,j,k) = utmp[m] - dssp*(rsd(m,i-2,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i+2,j,k));
			if (i == nx-3) for (m = 0; m < 5; m++) frct(m,nx-3,j,k) = utmp[m] - dssp*(rsd(m,nx-5,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);
			if (i == nx-2) for (m = 0; m < 5; m++) frct(m,nx-2,j,k) = utmp[m] - dssp*(rsd(m,nx-4,j,k)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);
		}

		i += RHSX_BLOCK-2;
	}
}

__global__ static void erhs_kernel_y (double *frct, const double *rsd, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u31;
	__shared__ double flux[RHSY_BLOCK][5];
	__shared__ double rtmp[RHSY_BLOCK*5];
	__shared__ double u21j[RHSY_BLOCK], u31j[RHSY_BLOCK], u41j[RHSY_BLOCK], u51j[RHSY_BLOCK];
	double utmp[5];;

	k = blockIdx.x+1;
	i = blockIdx.y+1;
	j = threadIdx.x;

	using namespace gpu_mod;

	while (j < ny) {
		// load u, rsd and rho_i using coalesced memory access along the m-axis
		// first compute number of threads executing this region 
		nthreads = ny-(j-threadIdx.x);
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, (j-threadIdx.x)+m/5, k);
		__syncthreads();

		//---------------------------------------------------------------------
		//   eta-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = rtmp[threadIdx.x*5+2];
		u31 = rtmp[threadIdx.x*5+2]/rtmp[threadIdx.x*5+0];
		q = 0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1] + rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2] + rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x][1] = rtmp[threadIdx.x*5+1]*u31;
		flux[threadIdx.x][2] = rtmp[threadIdx.x*5+2]*u31 + c2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][3] = rtmp[threadIdx.x*5+3]*u31;
		flux[threadIdx.x][4] = (c1*rtmp[threadIdx.x*5+4]-c2*q)*u31;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSY_BLOCK-1 && j < ny-1) 
			for (m = 0; m < 5; m++) utmp[m] = frct(m,i,j,k) - ty2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);
		u31 = 1.0/rtmp[threadIdx.x*5+0];
		u21j[threadIdx.x] = u31*rtmp[threadIdx.x*5+1];
		u31j[threadIdx.x] = u31*rtmp[threadIdx.x*5+2];
		u41j[threadIdx.x] = u31*rtmp[threadIdx.x*5+3];
		u51j[threadIdx.x] = u31*rtmp[threadIdx.x*5+4];

		__syncthreads();

		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = ty3*(u21j[threadIdx.x]-u21j[threadIdx.x-1]);
			flux[threadIdx.x][2] = (4.0/3.0)*ty3*(u31j[threadIdx.x]-u31j[threadIdx.x-1]);
			flux[threadIdx.x][3] = ty3*(u41j[threadIdx.x]-u41j[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*ty3*((u21j[threadIdx.x]*u21j[threadIdx.x]+u31j[threadIdx.x]*u31j[threadIdx.x]+u41j[threadIdx.x]*u41j[threadIdx.x]) - 
							(u21j[threadIdx.x-1]*u21j[threadIdx.x-1]+u31j[threadIdx.x-1]*u31j[threadIdx.x-1]+u41j[threadIdx.x-1]*u41j[threadIdx.x-1])) + 
							(1.0/6.0)*ty3*(u31j[threadIdx.x]*u31j[threadIdx.x]-u31j[threadIdx.x-1]*u31j[threadIdx.x-1]) + c1*c5*ty3*(u51j[threadIdx.x]-u51j[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSY_BLOCK-1 && j < ny-1) {
			utmp[0] += dy1*ty1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1] += ty3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dy2*ty1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2] += ty3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dy3*ty1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3] += ty3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dy4*ty1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4] += ty3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dy5*ty1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);

			//---------------------------------------------------------------------
			//   fourth-order dissipation
			//---------------------------------------------------------------------
			if (j == 1) for (m = 0; m < 5; m++) frct(m,i,1,k) = utmp[m] - dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,3,k));
			if (j == 2) for (m = 0; m < 5; m++) frct(m,i,2,k) = utmp[m] - dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,4,k));
			if (j >= 3 && j < ny-3) for (m = 0; m < 5; m++) frct(m,i,j,k) = utmp[m] - dssp*(rsd(m,i,j-2,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j+2,k));
			if (j == ny-3) for (m = 0; m < 5; m++) frct(m,i,ny-3,k) = utmp[m] - dssp*(rsd(m,i,ny-5,k)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);
			if (j == ny-2) for (m = 0; m < 5; m++) frct(m,i,ny-2,k) = utmp[m] - dssp*(rsd(m,i,ny-4,k)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);
		}

		j += RHSY_BLOCK-2;
	}
}

__global__ static void erhs_kernel_z (double *frct, const double *rsd, const int nx, const int ny, const int nz) {
	int i, j, k, m, nthreads;
	double q, u41;
	__shared__ double flux[RHSZ_BLOCK][5];
	__shared__ double rtmp[RHSZ_BLOCK*5];
	__shared__ double u21k[RHSZ_BLOCK], u31k[RHSZ_BLOCK], u41k[RHSZ_BLOCK], u51k[RHSZ_BLOCK];
	double utmp[5];

	j = blockIdx.x+1;
	i = blockIdx.y+1;
	k = threadIdx.x;
	
	using namespace gpu_mod;

	while (k < nz) {
		// load rsd using coalesced memory access along the m-axis
		// first compute number of threads executing this region 
		nthreads = (nz-(k-threadIdx.x));
		if (nthreads > blockDim.x) nthreads = blockDim.x;
		m = threadIdx.x;
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		m += nthreads;
		rtmp[m] = rsd(m%5, i, j, (k-threadIdx.x)+m/5);
		__syncthreads();

		//---------------------------------------------------------------------
		//   zeta-direction flux differences
		//---------------------------------------------------------------------
		flux[threadIdx.x][0] = rtmp[threadIdx.x*5+3];
		u41 = rtmp[threadIdx.x*5+3]/rtmp[threadIdx.x*5+0];
		q = 0.5*(rtmp[threadIdx.x*5+1]*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+2]*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+3]*rtmp[threadIdx.x*5+3])/rtmp[threadIdx.x*5+0];
		flux[threadIdx.x][1] = rtmp[threadIdx.x*5+1]*u41;
		flux[threadIdx.x][2] = rtmp[threadIdx.x*5+2]*u41;
		flux[threadIdx.x][3] = rtmp[threadIdx.x*5+3]*u41 + c2*(rtmp[threadIdx.x*5+4]-q);
		flux[threadIdx.x][4] = (c1*rtmp[threadIdx.x*5+4]-c2*q)*u41;
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSZ_BLOCK-1 && k < nz-1)
			for (m = 0; m < 5; m++) utmp[m] = frct(m,i,j,k) - tz2*(flux[threadIdx.x+1][m]-flux[threadIdx.x-1][m]);

		u41 = 1.0/rtmp[threadIdx.x*5+0];
		u21k[threadIdx.x] = u41*rtmp[threadIdx.x*5+1];
		u31k[threadIdx.x] = u41*rtmp[threadIdx.x*5+2];
		u41k[threadIdx.x] = u41*rtmp[threadIdx.x*5+3];
		u51k[threadIdx.x] = u41*rtmp[threadIdx.x*5+4];
		__syncthreads();

		if (threadIdx.x >= 1) {
			flux[threadIdx.x][1] = tz3*(u21k[threadIdx.x]-u21k[threadIdx.x-1]);
			flux[threadIdx.x][2] = tz3*(u31k[threadIdx.x]-u31k[threadIdx.x-1]);
			flux[threadIdx.x][3] = (4.0/3.0)*tz3*(u41k[threadIdx.x]-u41k[threadIdx.x-1]);
			flux[threadIdx.x][4] = 0.5*(1.0-c1*c5)*tz3*((u21k[threadIdx.x]*u21k[threadIdx.x]+u31k[threadIdx.x]*u31k[threadIdx.x]+u41k[threadIdx.x]*u41k[threadIdx.x]) - 
							(u21k[threadIdx.x-1]*u21k[threadIdx.x-1]+u31k[threadIdx.x-1]*u31k[threadIdx.x-1]+u41k[threadIdx.x-1]*u41k[threadIdx.x-1])) + 
							(1.0/6.0)*tz3*(u41k[threadIdx.x]*u41k[threadIdx.x]-u41k[threadIdx.x-1]*u41k[threadIdx.x-1]) + c1*c5*tz3*(u51k[threadIdx.x]-u51k[threadIdx.x-1]);
		}
		__syncthreads();

		if (threadIdx.x >= 1 && threadIdx.x < RHSZ_BLOCK-1 && k < nz-1) {
			utmp[0] += dz1*tz1*(rtmp[threadIdx.x*5-5]-2.0*rtmp[threadIdx.x*5+0]+rtmp[threadIdx.x*5+5]);
			utmp[1] += tz3*c3*c4*(flux[threadIdx.x+1][1]-flux[threadIdx.x][1]) + dz2*tz1*(rtmp[threadIdx.x*5-4]-2.0*rtmp[threadIdx.x*5+1]+rtmp[threadIdx.x*5+6]);
			utmp[2] += tz3*c3*c4*(flux[threadIdx.x+1][2]-flux[threadIdx.x][2]) + dz3*tz1*(rtmp[threadIdx.x*5-3]-2.0*rtmp[threadIdx.x*5+2]+rtmp[threadIdx.x*5+7]);
			utmp[3] += tz3*c3*c4*(flux[threadIdx.x+1][3]-flux[threadIdx.x][3]) + dz4*tz1*(rtmp[threadIdx.x*5-2]-2.0*rtmp[threadIdx.x*5+3]+rtmp[threadIdx.x*5+8]);
			utmp[4] += tz3*c3*c4*(flux[threadIdx.x+1][4]-flux[threadIdx.x][4]) + dz5*tz1*(rtmp[threadIdx.x*5-1]-2.0*rtmp[threadIdx.x*5+4]+rtmp[threadIdx.x*5+9]);

			//---------------------------------------------------------------------
			//   fourth-order dissipation
			//---------------------------------------------------------------------
			if (k == 1) for (m = 0; m < 5; m++) frct(m,i,j,1) = utmp[m] - dssp*(+5.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,3));
			if (k == 2) for (m = 0; m < 5; m++) frct(m,i,j,2) = utmp[m] - dssp*(-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,4));
			if (k >= 3 && k < nz-3) for (m = 0; m < 5; m++) frct(m,i,j,k) = utmp[m] - dssp*(rsd(m,i,j,k-2)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]+rsd(m,i,j,k+2));
			if (k == nz-3) for (m = 0; m < 5; m++) frct(m,i,j,nz-3) = utmp[m] - dssp*(rsd(m,i,j,nz-5)-4.0*rtmp[threadIdx.x*5+m-5]+6.0*rtmp[threadIdx.x*5+m]-4.0*rtmp[threadIdx.x*5+m+5]);
			if (k == nz-2) for (m = 0; m < 5; m++) frct(m,i,j,nz-2) = utmp[m] - dssp*(rsd(m,i,j,nz-4)-4.0*rtmp[threadIdx.x*5+m-5]+5.0*rtmp[threadIdx.x*5+m]);
		}

		k += RHSZ_BLOCK-2;
	}
}

//---------------------------------------------------------------------
//
//   compute the right hand side based on exact solution
//
//---------------------------------------------------------------------
void LU::erhs() {
	dim3 grid_full(nz,ny);
	dim3 grid_x(nz-2,ny-2);
	dim3 grid_y(nz-2,nx-2);
	dim3 grid_z(ny-2,nx-2);
	
	erhs_kernel_init<<<grid_full,nx>>>(frct, rsd, nx, ny, nz);
	erhs_kernel_x<<<grid_x, min(nx,RHSX_BLOCK)>>>(frct, rsd, nx, ny, nz);
	erhs_kernel_y<<<grid_y, min(ny,RHSY_BLOCK)>>>(frct, rsd, nx, ny, nz);
	erhs_kernel_z<<<grid_z, min(nz,RHSZ_BLOCK)>>>(frct, rsd, nx, ny, nz);
}

//---------------------------------------------------------------------
//   set up coefficients
//---------------------------------------------------------------------
void LU::setcoeff() {
	double dxi = 1.0/((double)nx-1.0);
	double deta = 1.0/((double)nx-1.0);
	double dzeta = 1.0/((double)nz-1.0);

	double tx1 = 1.0/(dxi*dxi);
	double tx2 = 1.0/(2.0*dxi);
	double tx3 = 1.0/dxi;

	double ty1 = 1.0/(deta*deta);
	double ty2 = 1.0/(2.0*deta);
	double ty3 = 1.0/deta;

	double tz1 = 1.0/(dzeta*dzeta);
	double tz2 = 1.0/(2.0*dzeta);
	double tz3 = 1.0/dzeta;

	//---------------------------------------------------------------------
	//   coefficients of the exact solution to the first pde
	//---------------------------------------------------------------------
	double ce[5*13];
	ce[0+0*5] = 2.0;
	ce[0+1*5] = 0.0;
	ce[0+2*5] = 0.0;
	ce[0+3*5] = 4.0;
	ce[0+4*5] = 5.0;
	ce[0+5*5] = 3.0;
	ce[0+6*5] = 0.5;
	ce[0+7*5] = 0.02;
	ce[0+8*5] = 0.01;
	ce[0+9*5] = 0.03;
	ce[0+10*5] = 0.5;
	ce[0+11*5] = 0.4;
	ce[0+12*5] = 0.3;

	//---------------------------------------------------------------------
	//   coefficients of the exact solution to the second pde
	//---------------------------------------------------------------------
	ce[1+0*5] = 1.0;
	ce[1+1*5] = 0.0;
	ce[1+2*5] = 0.0;
	ce[1+3*5] = 0.0;
	ce[1+4*5] = 1.0;
	ce[1+5*5] = 2.0;
	ce[1+6*5] = 3.0;
	ce[1+7*5] = 0.01;
	ce[1+8*5] = 0.03;
	ce[1+9*5] = 0.02;
	ce[1+10*5] = 0.4;
	ce[1+11*5] = 0.3;
	ce[1+12*5] = 0.5;

	//---------------------------------------------------------------------
	//   coefficients of the exact solution to the third pde
	//---------------------------------------------------------------------
	ce[2+0*5] = 2.0;
	ce[2+1*5] = 2.0;
	ce[2+2*5] = 0.0;
	ce[2+3*5] = 0.0;
	ce[2+4*5] = 0.0;
	ce[2+5*5] = 2.0;
	ce[2+6*5] = 3.0;
	ce[2+7*5] = 0.04;
	ce[2+8*5] = 0.03;
	ce[2+9*5] = 0.05;
	ce[2+10*5] = 0.3;
	ce[2+11*5] = 0.5;
	ce[2+12*5] = 0.4;

	//---------------------------------------------------------------------
	//   coefficients of the exact solution to the fourth pde
	//---------------------------------------------------------------------
	ce[3+0*5] = 2.0;
	ce[3+1*5] = 2.0;
	ce[3+2*5] = 0.0;
	ce[3+3*5] = 0.0;
	ce[3+4*5] = 0.0;
	ce[3+5*5] = 2.0;
	ce[3+6*5] = 3.0;
	ce[3+7*5] = 0.03;
	ce[3+8*5] = 0.05;
	ce[3+9*5] = 0.04;
	ce[3+10*5] = 0.2;
	ce[3+11*5] = 0.1;
	ce[3+12*5] = 0.3;

	//---------------------------------------------------------------------
	//   coefficients of the exact solution to the fifth pde
	//---------------------------------------------------------------------
	ce[4+0*5] = 5.0;
	ce[4+1*5] = 4.0;
	ce[4+2*5] = 3.0;
	ce[4+3*5] = 2.0;
	ce[4+4*5] = 0.1;
	ce[4+5*5] = 0.4;
	ce[4+6*5] = 0.3;
	ce[4+7*5] = 0.05;
	ce[4+8*5] = 0.04;
	ce[4+9*5] = 0.03;
	ce[4+10*5] = 0.1;
	ce[4+11*5] = 0.3;
	ce[4+12*5] = 0.2;

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dxi, &dxi, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::deta, &deta, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dzeta, &dzeta, sizeof(double)));

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx1, &tx1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx2, &tx2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tx3, &tx3, sizeof(double)));

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty1, &ty1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty2, &ty2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ty3, &ty3, sizeof(double)));

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz1, &tz1, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz2, &tz2, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::tz3, &tz3, sizeof(double)));

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::ce, &ce, 13*5*sizeof(double)));

	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::dt, &dt, sizeof(double)));
	HANDLE_ERROR (cudaMemcpyToSymbol (gpu_mod::omega, &omega, sizeof(double)));
}

void LU::allocate_device_memory() {
	int gridsize = nx*ny*nz;
	int norm_buf_size = max(5*(ny-2)*(nz-2), ((nx-3)*(ny-3)+(nx-3)*(nz-3)+(ny-3)*(nz-3))/((PINTGR_BLOCK-1)*(PINTGR_BLOCK-1))+3);

	HANDLE_ERROR(cudaMalloc((void **)&u, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rsd, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&frct, 5*gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&rho_i, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&qs, gridsize*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void **)&dev_norm_buf, norm_buf_size*sizeof(double)));
}

void LU::free_device_memory() {
	HANDLE_ERROR(cudaFree(u));
	HANDLE_ERROR(cudaFree(rsd));
	HANDLE_ERROR(cudaFree(frct));
	HANDLE_ERROR(cudaFree(rho_i));
	HANDLE_ERROR(cudaFree(qs));
	HANDLE_ERROR(cudaFree(dev_norm_buf));
}

void LU::get_cuda_info() {
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
