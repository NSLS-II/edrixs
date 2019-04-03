.SUFFIXES: .f90

include ./make.sys

objects=m_constants.o m_control.o m_types.o m_global.o utils.o io.o fock.o spmv.o linsys.o  ham.o  full_diag.o lanczos.o  arpack.o ed_driver.o xas_driver.o rixs_driver.o opavg_driver.o

default: all

all: ed xas rixs opavg

ed: $(objects) ed_main.o 
	$(LINKER) $(objects) ed_main.o -o ed.x  $(LIBS)

xas: $(objects) xas_main.o
	$(LINKER) $(objects) xas_main.o -o xas.x  $(LIBS)

rixs: $(objects) rixs_main.o
	$(LINKER) $(objects) rixs_main.o -o rixs.x  $(LIBS)

opavg: $(objects) opavg_main.o
	$(LINKER) $(objects) opavg_main.o -o opavg.x  $(LIBS)

.f90.o:
	$(F90) $(FFLAGS) $*.f90

install:
	cp *.x ../bin 	

clean:
	rm -f *.mod
	rm -f *.o
	rm -f *.x
	rm -f ../bin/ed.x
	rm -f ../bin/xas.x
	rm -f ../bin/rixs.x
	rm -f ../bin/opavg.x
