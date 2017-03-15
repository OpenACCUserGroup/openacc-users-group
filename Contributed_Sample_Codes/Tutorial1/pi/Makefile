choose:
	@echo make c++ or make fortran or make both

CPP=pgc++
FC=pgfortran
TIMER=/usr/bin/time
OPT=
NOPT=-fast -Minfo=opt $(OPT)
STEPS=1000

c++: pic.exe
	$(TIMER) ./pic.exe $(STEPS)
pic.exe: pic.cpp
	$(CPP) -o $@ $< $(NOPT)

fortran: pif.exe
	$(TIMER) ./pif.exe $(STEPS)
pif.exe: pif.f90
	$(FC) -o $@ $< $(NOPT)

both: c++ fortran

clean:
	rm -f *.o *.exe *.s a.out
