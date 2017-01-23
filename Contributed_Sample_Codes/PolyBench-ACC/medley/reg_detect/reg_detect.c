/* POLYBENCH/GPU-OPENACC
 *
 * This file is a part of the Polybench/GPU-OpenACC suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is int, default size is 50. */
#include "reg_detect.h"


/* Array initialization. */
static
void init_array(int maxgrid,
		DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++) {
      sum_tang[i][j] = (DATA_TYPE)((i+1)*(j+1));
      mean[i][j] = ((DATA_TYPE) i-j) / maxgrid;
      path[i][j] = ((DATA_TYPE) i*(j-1)) / maxgrid;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int maxgrid,
		 DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid))
{
  int i, j;

  for (i = 0; i < maxgrid; i++)
    for (j = 0; j < maxgrid; j++) {
      fprintf (stderr, DATA_PRINTF_MODIFIER, path[i][j]);
      if ((i * maxgrid + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Source (modified): http://www.cs.uic.edu/~iluican/reg_detect.c */
static
void kernel_reg_detect(int niter, int maxgrid, int length,
		       DATA_TYPE POLYBENCH_2D(sum_tang,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(mean,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_2D(path,MAXGRID,MAXGRID,maxgrid,maxgrid),
		       DATA_TYPE POLYBENCH_3D(diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length),
		       DATA_TYPE POLYBENCH_3D(sum_diff,MAXGRID,MAXGRID,LENGTH,maxgrid,maxgrid,length))
{
  int t, i, j, cnt;

  #pragma scop
  #pragma acc data copy(path) copyin(sum_tang,mean) create(diff,sum_diff)
  {
    #pragma acc parallel
    {
      for (t = 0; t < _PB_NITER; t++)
	{
	  #pragma acc loop
	  for (j = 0; j <= _PB_MAXGRID - 1; j++)
	    for (i = j; i <= _PB_MAXGRID - 1; i++)
	      #pragma acc loop
	      for (cnt = 0; cnt <= _PB_LENGTH - 1; cnt++)
		diff[j][i][cnt] = sum_tang[j][i];
	  #pragma acc loop
	  for (j = 0; j <= _PB_MAXGRID - 1; j++)
	    {
	      for (i = j; i <= _PB_MAXGRID - 1; i++)
		{
		  sum_diff[j][i][0] = diff[j][i][0];
		  #pragma acc loop
		  for (cnt = 1; cnt <= _PB_LENGTH - 1; cnt++)
		    sum_diff[j][i][cnt] = sum_diff[j][i][cnt - 1] + diff[j][i][cnt];
		  mean[j][i] = sum_diff[j][i][_PB_LENGTH - 1];
		}
	    }
	  
	  #pragma acc loop
	  for (i = 0; i <= _PB_MAXGRID - 1; i++)
	    path[0][i] = mean[0][i];
	  #pragma acc loop
	  for (j = 1; j <= _PB_MAXGRID - 1; j++)
	    #pragma acc loop
	    for (i = j; i <= _PB_MAXGRID - 1; i++)
	      path[j][i] = path[j - 1][i - 1] + mean[j][i];
	}
    }
  }
  #pragma endscop
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int niter = NITER;
  int maxgrid = MAXGRID;
  int length = LENGTH;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(sum_tang, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_2D_ARRAY_DECL(mean, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, MAXGRID, MAXGRID, maxgrid, maxgrid);
  POLYBENCH_3D_ARRAY_DECL(diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);
  POLYBENCH_3D_ARRAY_DECL(sum_diff, DATA_TYPE, MAXGRID, MAXGRID, LENGTH, maxgrid, maxgrid, length);
  
  /* Initialize array(s). */
  init_array (maxgrid,
	      POLYBENCH_ARRAY(sum_tang),
	      POLYBENCH_ARRAY(mean),
	      POLYBENCH_ARRAY(path));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_reg_detect (niter, maxgrid, length,
		     POLYBENCH_ARRAY(sum_tang),
		     POLYBENCH_ARRAY(mean),
		     POLYBENCH_ARRAY(path),
		     POLYBENCH_ARRAY(diff),
		     POLYBENCH_ARRAY(sum_diff));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(maxgrid, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(sum_tang);
  POLYBENCH_FREE_ARRAY(mean);
  POLYBENCH_FREE_ARRAY(path);
  POLYBENCH_FREE_ARRAY(diff);
  POLYBENCH_FREE_ARRAY(sum_diff);

  return 0;
}
