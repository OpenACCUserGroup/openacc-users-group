PROGRAM=sp
CPP=g++
CPPFLAGS=-O3 -c -Wall -g
CLINKFLAGS=
CLIB=
NVCOMPILE=nvcc
NVFLAGS=-O3 -c -g -arch=sm_35
NVLIB=

OBJS = sp.o timers.o sp_kernels.o

${PROGRAM}: ${OBJS}
	${NVCOMPILE} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CLIB} ${NVLIB}

.cpp.o:
	${CPP} ${CPPFLAGS} $<
%.o: %.cu 
	${NVCOMPILE} ${NVFLAGS} $<

sp.o:		sp.cpp main.h
timers.o:	timers.cpp main.h
sp_kernels.o:	sp_kernels.cu main.h

clean:
	- rm -f *.o 
	- rm -f core
	- rm -f ${PROGRAM}
