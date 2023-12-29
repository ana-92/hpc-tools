# Default Lapacke: Openblas at CESGA
LDLIBS=-lopenblas

# Other systems (my Debian boxes, for example)
# LDLIBS=-llapacke

# Intel MKL at CESGA
# Module needed: imkl
# => module load openblas
# LDLIBS for intel compiler: icx (module needed: intel)
# Just invoke make like this: make CC=icx CFLAGS="-Wall -O2 -qopt-report=5"
#LDLIBS=-qmkl=sequential -lmkl_intel_lp64
SIZE = 500 

CC ?= gcc
CFLAGS ?= -Wall -O2 -qopt-report=5
COMPILER_NAME := $(CC)
OPTIMIZATION_LEVEL := $(subst -O,,$(filter -O%,$(CFLAGS)))

EXECUTABLE = my_dgesv_$(COMPILER_NAME)_$(OPTIMIZATION_LEVEL).exe

$(EXECUTABLE): dgesv.o timer.o main.o
	$(CC) $(CFLAGS) -o $(EXECUTABLE) dgesv.o timer.o main.o $(LDLIBS) -lm

clean:
	$(RM) dgesv *.o *~

run:
	./dgesv $(SIZE)
