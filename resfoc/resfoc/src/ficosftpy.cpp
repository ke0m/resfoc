/**
 * Python interface for cosine transform
 * @author: Joseph Jennings
 * @version: 2020.04.02
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ficosft.h"

namespace py = pybind11;

PYBIND11_MODULE(ficosft,m) {
  m.doc() = "Forward and inverse cosine Fourier transform";
  m.def("fwdcosft",[](
      int dim1,
      int n1,
      int n2,
      py::array_t<int, py::array::c_style> n,
      py::array_t<int, py::array::c_style> sign,
      py::array_t<int, py::array::c_style> s,
      py::array_t<float, py::array::c_style> data,
      bool verb
      )
      {
        fwdcosft(dim1,n1,n2,n.mutable_data(),sign.mutable_data(),s.mutable_data(),data.mutable_data(),verb);
      },
      py::arg("dim1"), py::arg("n1"), py::arg("n2"), py::arg("n"),
      py::arg("sign"), py::arg("s"), py::arg("data"), py::arg("verb")
      );
  m.def("invcosft",[](
      int dim1,
      int n1,
      int n2,
      py::array_t<int, py::array::c_style> n,
      py::array_t<int, py::array::c_style> sign,
      py::array_t<int, py::array::c_style> s,
      py::array_t<float, py::array::c_style> data,
      bool verb
      )
      {
        invcosft(dim1,n1,n2,n.mutable_data(),sign.mutable_data(),s.mutable_data(),data.mutable_data(),verb);
      },
      py::arg("dim1"), py::arg("n1"), py::arg("n2"), py::arg("n"),
      py::arg("sign"), py::arg("s"), py::arg("data"), py::arg("verb")
  );
}
