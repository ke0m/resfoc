/**
 * Python interface to the depth to time
 * conversion function
 * @author: Joseph Jennings
 * @version: 2020.02.25
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "depth2time.h"

namespace py = pybind11;

PYBIND11_MODULE(depth2time,m) {
  m.doc() = "Conversion of depth images to time";
  m.def("convert2time",[](
      int nh,
      int nm,
      int nz,
      float oz,
      float dz,
      int nt,
      float ot,
      float dt,
      py::array_t<float, py::array::c_style> vel,
      py::array_t<float, py::array::c_style> depth,
      py::array_t<float, py::array::c_style> time
      )
      {
        convert2time(nh, nm, nz, oz, dz, nt, ot, dt, vel.mutable_data(), depth.mutable_data(), time.mutable_data());
      },
      py::arg("nh"), py::arg("nm"), py::arg("nz"), py::arg("oz"), py::arg("dz"),
      py::arg("nt"), py::arg("ot"), py::arg("dt"),
      py::arg("vel"), py::arg("depth"), py::arg("time")
      );
}
