PROGRAM=lu
CPP=g++
CPPFLAGS=-O3 -c -Wall -g
CLINKFLAGS=
CLIB=
NVCOMPILE=nvcc
NVFLAGS=-O3 -c -g -arch=sm_35 
NVLIB=

OBJS = lu.o timers.o lu_kernels.o

${PROGRAM}: ${OBJS}
	${NVCOMPILE} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CLIB} ${NVLIB}

.cpp.o:
	${CPP} ${CPPFLAGS} $<
%.o: %.cu 
	${NVCOMPILE} ${NVFLAGS} $<

lu.o:		lu.cpp main.h
timers.o:	timers.cpp main.h
lu_kernels.o:	lu_kernels.cu main.h

clean:
	- rm -f *.o 
	- rm -f core
	- rm -f ${PROGRAM}
