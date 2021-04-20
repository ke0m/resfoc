/**
 * Python interface to pick function
 * @author: Joseph Jennings
 * @version: 2020.06.19
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "pickscan.h"

namespace py = pybind11;

PYBIND11_MODULE(pickscan,m) {
  m.doc() = "Computes shifts for refocusing rho";
  m.def("pickscan",[](
      float an,
      int gate,
      bool norm,
      float vel0,
      float o2,
      float d2,
      int n1,
      int n2,
      int n3,
      py::array_t<float, py::array::c_style> allscn,
      py::array_t<float, py::array::c_style> pck2,
      py::array_t<float, py::array::c_style> ampl,
      py::array_t<float, py::array::c_style> pcko
      )
      {
        pickscan(an,gate,norm,vel0,o2,d2,n1,n2,n3,allscn.mutable_data(),
             pck2.mutable_data(),ampl.mutable_data(),pcko.mutable_data());
      },
      py::arg("an"), py::arg("gate"), py::arg("norm"), py::arg("vel0"), py::arg("o2"),
      py::arg("d2"), py::arg("n1"), py::arg("n2"), py::arg("n3"), py::arg("allscn"),
      py::arg("pck2"), py::arg("ampl"), py::arg("pcko")
      );
}
