%.o: %.c
	gcc -O3 -fPIC -ffast-math -fopenmp -shared -Wall \
		-I. -I../../external_cpp/ -c -o $@ $<

%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall \
		-I. -I../../external_cpp/ -c -fPIC -o $@ $<

INSTALL_DIR=..

default: depth2time rstolt rstoltbig rhoshifts ficosft pickscan

depth2time: depth2time.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g stretch.o depth2time.o depth2timepy.cpp -o \
		$(INSTALL_DIR)/depth2time`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

rstolt: rstolt.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g stretch.o rstolt.o rstoltpy.cpp -o \
		$(INSTALL_DIR)/rstolt`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

rstoltbig: rstoltbig.o cosft.o ficosft.o kiss_fft.o kiss_fftr.o stretch.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g stretch.o kiss_fft.o kiss_fftr.o cosft.o ficosft.o rstoltbig.o rstoltbigpy.cpp \
		-o $(INSTALL_DIR)/rstoltbig`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

rhoshifts: rhoshifts.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g rhoshifts.o rhoshiftspy.cpp -o \
		$(INSTALL_DIR)/rhoshifts`${CONDA_PREFIX}/bin/python3-config --extension-suffix`

ficosft: cosft.o ficosft.o kiss_fft.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g kiss_fft.o cosft.o ficosft.o ficosftpy.cpp -o \
		$(INSTALL_DIR)/ficosft`${CONDA_PREFIX}/bin/python3-config --extension-suffix` -lfftw3f

pickscan: dynprog.o pickscan.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
    -g dynprog.o pickscan.o pickscanpy.cpp -o \
		$(INSTALL_DIR)/pickscan`${CONDA_PREFIX}/bin/python3-config --extension-suffix`
