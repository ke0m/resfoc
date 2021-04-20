/**
 * Python interface to rho shifts function
 * @author: Joseph Jennings
 * @version: 2020.04.29
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "rhoshifts.h"

namespace py = pybind11;

PYBIND11_MODULE(rhoshifts,m) {
  m.doc() = "Computes shifts for refocusing rho";
  m.def("rhoshifts",[](
      int nro,
      int nx,
      int nz,
      float dro,
      py::array_t<float, py::array::c_style> rho,
      py::array_t<float, py::array::c_style> coords
      )
      {
        rhoshifts(nro, nx, nz, dro, rho.mutable_data(), coords.mutable_data());
      },
      py::arg("nro"), py::arg("nx"), py::arg("nz"),
      py::arg("dro"), py::arg("rho"), py::arg("coords")
      );
}
