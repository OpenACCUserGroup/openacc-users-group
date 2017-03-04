SHELL=/bin/sh
BENCHMARK=ua
BENCHMARKU=UA

include ../config/make.def

include ../sys/make.common

OBJS = ua.o convect.o diffuse.o adapt.o move.o mason.o \
       precond.o utils.o verify.o setup.o \
       ${COMMON}/print_results.o ${COMMON}/c_timers.o ${COMMON}/wtime.o


# npbparams.h is included by header.h
# The following rule should do the trick but many make programs (not gmake)
# will do the wrong thing and rebuild the world every time (because the
# mod time on header.h is not changed. One solution would be to 
# touch header.h but this might cause confusion if someone has
# accidentally deleted it. Instead, make the dependency on npbparams.h
# explicit in all the lines below (even though dependence is indirect). 

# header.h: npbparams.h

${PROGRAM}: config ${OBJS}
	@if [ x$(UPDATE) = xau ] ; then			\
		${MAKE} PROGRAM=${PROGRAM} ua-au;	\
	else						\
		${MAKE} PROGRAM=${PROGRAM} ua-lk;	\
	fi

ua-lk: ${OBJS} transfer.o
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} transfer.o ${C_LIB}

ua-au: ${OBJS} transfer_au.o
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM}.au ${OBJS} transfer_au.o ${C_LIB}

.c.o:
	${CCOMPILE} $<

ua.o:        ua.c       header.h npbparams.h
setup.o:     setup.c    header.h npbparams.h
convect.o:   convect.c  header.h npbparams.h
adapt.o:     adapt.c    header.h npbparams.h
move.o:      move.c     header.h npbparams.h
diffuse.o:   diffuse.c  header.h npbparams.h
mason.o:     mason.c    header.h npbparams.h
precond.o:   precond.c  header.h npbparams.h
transfer.o:  transfer.c header.h npbparams.h
transfer_au.o:  transfer_au.c header.h npbparams.h
utils.o:     utils.c    header.h npbparams.h
verify.o:    verify.c   header.h npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
