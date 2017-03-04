PROGRAM=bt
CPP=g++
CPPFLAGS=-O3 -c -Wall -g
CLINK=g++
CLINKFLAGS=
CLIB=
NVCOMPILE=nvcc
NVFLAGS=-O3 -c -g -arch=sm_35 
NVLIB=

OBJS = bt.o timers.o bt_kernels.o

${PROGRAM}: ${OBJS}
	${NVCOMPILE} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${CLIB} ${NVLIB}

.cpp.o:
	${CPP} ${CPPFLAGS} $<
%.o: %.cu 
	${NVCOMPILE} ${NVFLAGS} $<

bt.o:		bt.cpp main.h
timers.o:	timers.cpp main.h
bt_kernels.o:	bt_kernels.cu main.h

clean:
	- rm -f *.o 
	- rm -f core
	- rm -f ${PROGRAM}
