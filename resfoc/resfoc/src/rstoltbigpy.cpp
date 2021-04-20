/**
 * Python interface to rstolt.cpp
 * @author: Joseph Jennings
 * @version: 2019.12.14
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rstoltbig.h"

namespace py = pybind11;

PYBIND11_MODULE(rstoltbig,m) {
  m.doc() = "Stolt residual migration for large images";

  py::class_<rstoltbig>(m,"rstoltbig")
      .def(py::init<int,int,int,int,int,int,int,float,float,float,float,float>(),
          py::arg("nz"),  py::arg("nm"),  py::arg("nh"),
          py::arg("nzp"), py::arg("nmp"), py::arg("nhp"), py::arg("nro"),
          py::arg("dz"), py::arg("dm"), py::arg("dh"), py::arg("dro"),
          py::arg("oro"))
      .def("resmig",[](rstoltbig &rst,
          py::array_t<float, py::array::c_style> dat,
          py::array_t<float, py::array::c_style> img,
          int nthrd,
          bool verb
          )
          {
            rst.resmig(dat.mutable_data(), img.mutable_data(), nthrd, verb);
          },
          py::arg("dat"), py::arg("img"), py::arg("nthrd"), py::arg("verb")
      );
}
