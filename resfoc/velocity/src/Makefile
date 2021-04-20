%.o: %.cpp
	g++ -std=c++1y -O3 -fPIC -ffast-math -fopenmp -shared -Wall \
		-I. -I../../external_cpp/ -c -fPIC -o $@ $<

INSTALL_DIR=..

default: evntcre8

evntcre8: evntcre8.o
	g++ -O3 -fPIC -ffast-math -fopenmp -std=c++1y -shared -Wall \
		`${CONDA_PREFIX}/bin/python3 -m pybind11 --includes` \
		-g evntcre8.o evntcre8py.cpp -o \
		$(INSTALL_DIR)/evntcre8`${CONDA_PREFIX}/bin/python3-config --extension-suffix` \
		-ltbb -lfftw3f -lfftw3f_threads
