SHELL=/bin/sh
BENCHMARK=sp
BENCHMARKU=SP
DEFINES=-DCRPL_COMP=0
PXM=acc

include ../config/make.def

include ../sys/make.common

OBJS = sp.o initialize.o exact_solution.o exact_rhs.o \
       set_constants.o adi.o rhs.o      \
	   add.o txinvr.o error.o verify.o \
       print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o 

# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
	${CCOMPILE} ${DEFINES} $<

sp.o:             sp.c  header.h npbparams.h
initialize.o:     initialize.c  header.h npbparams.h
exact_solution.o: exact_solution.c  header.h npbparams.h
exact_rhs.o:      exact_rhs.c  header.h npbparams.h
set_constants.o:  set_constants.c  header.h npbparams.h
adi.o:            adi.c  header.h npbparams.h
rhs.o:            rhs.c  header.h npbparams.h
#lhsx.o:           lhsx.c  header.h npbparams.h
#lhsy.o:           lhsy.c  header.h npbparams.h
#lhsz.o:           lhsz.c  header.h npbparams.h
x_solve.o:        x_solve.c  header.h npbparams.h
ninvr.o:          ninvr.c  header.h npbparams.h
y_solve.o:        y_solve.c  header.h npbparams.h
pinvr.o:          pinvr.c  header.h npbparams.h
z_solve.o:        z_solve.c  header.h npbparams.h
#tzetar.o:         tzetar.c  header.h npbparams.h
add.o:            add.c  header.h npbparams.h
txinvr.o:         txinvr.c  header.h npbparams.h
error.o:          error.c  header.h npbparams.h
verify.o:         verify.c  header.h npbparams.h

clean:
	- rm -f *.x *.o *~ mputil* ../common/*.o *.i *.cu *.ptx *.w2c.c *.w2c.h *.t *.B *.spin
	- rm -f npbparams.h core
