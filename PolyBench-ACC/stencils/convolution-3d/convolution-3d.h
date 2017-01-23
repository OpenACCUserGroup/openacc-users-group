/**
 * adi.h: This file is part of the PolyBench/C 3.2 test suite.
 *
 *
 * Contact: Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://polybench.sourceforge.net
 */
#ifndef CONV3D_H
#define CONV3D_H

/* Default to LARGE_DATASET. */
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET)
#  define LARGE_DATASET
# endif

/* Do not define anything if the user manually defines the size. */
# if !defined(NI) && ! defined(NJ) && ! defined(NK)
/* Define the possible dataset sizes. */
#  ifdef MINI_DATASET
#   define NI 64
#   define NJ 64
#   define NK 64
#  endif

#  ifdef SMALL_DATASET
#   define NI 128
#   define NJ 128
#   define NK 128
#  endif

#  ifdef STANDARD_DATASET
#   define NI 192
#   define NJ 192
#   define NK 192
#  endif

#  ifdef LARGE_DATASET /* Default if unspecified. */
#   define NI 256
#   define NJ 256
#   define NK 256
#  endif

#  ifdef EXTRALARGE_DATASET
#   define NI 384
#   define NJ 384
#   define NK 384
#  endif
# endif /* !N */

# define _PB_NI POLYBENCH_LOOP_BOUND(NI,ni)
# define _PB_NJ POLYBENCH_LOOP_BOUND(NJ,nj)
# define _PB_NK POLYBENCH_LOOP_BOUND(NK,nk)

# ifndef DATA_TYPE
#  define DATA_TYPE float
#  define DATA_PRINTF_MODIFIER "%0.2lf "
# endif


#endif /* !CONV3D */
