! Fortran program to estimate value of pi
! See discussion for Improved Say of Determining Pi at
!  http://mb-soft.com/public3/pi.html  (retrieved March 2017)
! If we integrate 1/(1+x^2) for x=0:1, we get pi/4

program piprogram
  implicit none
  integer(8) :: i, nsteps
  doubleprecision :: pi, step, sum, x
  character*10 :: cnsteps
  nsteps = 0
  if (command_argument_count() > 0) then
    call get_command_argument(1, cnsteps)
    read(cnsteps,'(i)') nsteps
  endif
  if (nsteps <= 0) nsteps = 100
  step = 1.0d0 / nsteps
  do i = 1, nsteps
    x = (i - 0.5) * step
    sum = sum + 1.0 / (1.0 + x*x)
  enddo
  pi = 4.0 * step * sum
  print '(a,f20.17)', 'pi is ', pi
end program
