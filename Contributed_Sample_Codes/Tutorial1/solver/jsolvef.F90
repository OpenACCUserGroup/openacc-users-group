! Jacobi iterative method for solving a system of linear equations
! This is guaranteed to converge if the matrix is diagonally dominant,
! so we artificially force the matrix to be diagonally dominant.
!  See https://en.wikipedia.org/wiki/Jacobi_method
!
!  We solve for vector x in Ax = b
!    Rewrite the matrix A as a
!       lower triangular (L),
!       upper triangular (U),
!    and diagonal matrix (D).
!
!  Ax = (L + D + U)x = b
!
!  rearrange to get: Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D
!
!  we can do this iteratively: x_new = (b-(L+U)x_old)/D

! build with TYPESIZE=8 (default) or TYPESIZE=4
! build with TOLERANCE=0.001 (default) or TOLERANCE= any other value
! three arguments:
!   vector size
!   maximum iteration count
!   frequency of printing the residual (every n-th iteration)

#ifndef TYPESIZE
#define TYPESIZE 8
#endif

#define TOLERANCE 0.001

module jinit
  implicit none
  integer, parameter :: sz = TYPESIZE
contains
  subroutine init_simple_diag_dom(A)
    real(sz), dimension(:,:) :: A
    integer i,j,nsize
    real(sz) :: sum, x

    nsize = ubound(A,1)

    ! In a diagonally-dominant matrix, the diagonal element
    ! is greater than the sum of the other elements in the row.
    ! Scale the matrix so the sum of the row elements is close to one.

    do i = 1, nsize
      sum = 0
      do j = 1, nsize
        call random_number(x)
        x = mod(x, 23.0_sz) / 1000.0_sz
        A(j,i) = x
        sum = sum + x
      enddo
      A(i,i) = A(i,i) + sum
      ! scale the row so the final matrix is almost an identity matrix
      do j = 1, nsize
        A(j,i) = A(j,i) / sum
      enddo
    enddo
  end subroutine
end module

program main
  use jinit
  use omp_lib
  implicit none
  integer :: nsize, i, j, iters, max_iters, riter
  double precision :: start_time, elapsed_time
  real(sz), allocatable :: A(:,:), b(:)
  real(sz), allocatable, target :: x1(:), x2(:)
  real(sz), pointer, contiguous :: xnew(:), xold(:), xtmp(:)
  real(sz) :: r, residual, rsum, dif, err, chksum
  character*10 :: cnsteps

  ! set matrix dimensions and allocate memory for matrices
  nsize = 0
  if (command_argument_count() > 0) then
    call get_command_argument(1, cnsteps)
    read(cnsteps,'(i)') nsize
  endif
  if (nsize <= 0) nsize = 1000

  max_iters = 0
  if (command_argument_count() > 1) then
    call get_command_argument(2, cnsteps)
    read(cnsteps,'(i)') max_iters
  endif
  if (max_iters <= 0) max_iters = 5000

  riter = 0
  if (command_argument_count() > 2) then
    call get_command_argument(3, cnsteps)
    read(cnsteps,'(i)') riter
  endif
  if (riter <= 0) riter = 200

  print *, 'nsize = ', nsize, ', max_iters = ', max_iters

  allocate(A(nsize,nsize))
  allocate(b(nsize), x1(nsize), x2(nsize))

  ! generate a diagonally dominant matrix
  call init_simple_diag_dom(A)

  ! zero the x vectors, random values to the b vector
  x1 = 0
  x2 = 0
  do i = 1, nsize
    call random_number(r)
    b(i) = mod(r, 51.0_sz) / 100.0_sz
  enddo

  start_time = omp_get_wtime()

  !
  ! jacobi iterative solver
  !

  residual = TOLERANCE + 1.0
  iters = 0
  xnew => x1	! swap these pointers in each iteration
  xold => x2
  do while(residual > TOLERANCE .and. iters < max_iters)
    iters = iters + 1
    ! swap input and output vectors
    xtmp => xnew
    xnew => xold
    xold => xtmp

    do i = 1, nsize
      rsum = 0
      do j = 1, nsize
        if( i /= j ) rsum = rsum + A(j,i) * xold(j)
      enddo
      xnew(i) = (b(i) - rsum) / A(i,i)
    enddo
    !
    ! test convergence, sqrt(sum((xnew-xold)**2))
    !
    residual = 0
    do i = 1, nsize
      dif = xnew(i) - xold(i)
      residual = residual + dif * dif
    enddo
    residual = sqrt(residual)
    if( mod(iters, riter) == 0 ) print *, 'Iteration ', iters, ',&
               & residual is ', residual
  enddo
  elapsed_time = omp_get_wtime() - start_time
  print *, 'Converged after ', iters, ' iterations'
  print *, '            and ', elapsed_time, ' seconds'
  print *, '    residual is ', residual

  !
  ! test answer by multiplying my computed value of x by
  ! the input A matrix and comparing the result with the
  ! input b vector.
  !
  err = 0
  chksum = 0

  do i = 1, nsize
    rsum = 0
    do j = 1, nsize
      rsum = rsum + A(j,i) * xnew(j)
    enddo
    rsum = rsum - b(i)
    err = err + rsum*rsum
    chksum = chksum + xnew(i)
  enddo
  err = sqrt(err)
  print *, 'Solution error is ', err
  print *, 'chksum is ', chksum
  if (err > TOLERANCE) then
    print *, '****** Final Solution Out of Tolerance ******'
    print *, err, ' > ', TOLERANCE
  endif
  deallocate(A, b, x1, x2)
end program
