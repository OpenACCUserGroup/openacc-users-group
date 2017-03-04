# === Basics ===
#CC       = gcc
#CXX      = g++
#LD       = g++
#AR       = ar
#RANLIB   = ranlib

# In CPPFLAGS, note src/common is from the SHOC source tree, so we must
# use $(srcdir).  In contrast, the files in config used in the build are
# generated as part of the configuration, so we want to find them in the
# build tree - hence we do not use $(srcdir) for that -I specification.
#CPPFLAGS += -I$(top_srcdir)/src/common -I$(top_builddir)/config 
CPPFLAGS += -I$(top_srcdir)/src/common -I$(top_builddir)/config
#CFLAGS   += -g -O2
#CXXFLAGS += -g -O2
NVCXXFLAGS = -g -O2
#ARFLAGS  = rcv
#LDFLAGS  =  -L$(top_builddir)/src/common
LDFLAGS  += -L$(top_builddir)/src/common
LIBS     = 

USE_MPI         = no
MPICXX          = 
MPI_CPPFLAGS	= -DPARALLEL

OCL_CPPFLAGS    = -I$(top_srcdir)/src/opencl/common
OCL_LDFLAGS		= -L$(top_builddir)/src/opencl/common
OCL_LIBS        = -lSHOCCommonOpenCL -lSHOCCommon -framework OpenCL

NVCC            = 
CUDA_CXX        = 
CUDA_INC        = -I -I$(top_srcdir)/src/cuda/common
CUDA_LDFLAGS	= -L$(top_builddir)/src/cuda/common
CUDA_CPPFLAGS   =  -I$(top_srcdir)/src/cuda/common

USE_CUDA        = no
ifeq ($(USE_CUDA),yes)
CUDA_LIBS		:= -lSHOCCommon $(shell $(top_srcdir)/config/find_cuda_libs.sh )
else
CUDA_LIBS       =
endif

OPENACC_CPPFLAGS	= -I$(top_srcdir)/src/openacc/common


