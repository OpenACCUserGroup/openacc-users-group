SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG
DEFINES=-DCRPL_COMP=0
PXM=acc

include ../config/make.def

include ../sys/make.common

OBJS = cg.o \
       ${COMMON}/print_results.o  \
       ${COMMON}/${RAND}.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/wtime.o


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} ${DEFINES} $<

cg.o:		cg.c  globals.h npbparams.h

clean:
	- rm -f *.x *.w2c.ptx *.o *.w2c.cu *.w2c.c *.w2c.h *.i *.spin *.B *.s *.t *~ ../common/*.o
	- rm -f npbparams.h core



