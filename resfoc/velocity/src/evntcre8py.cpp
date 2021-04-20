/**
 * Python interface to C++ geologic event creator
 * @author: Joseph Jennings
 * @version: 2020.04.22
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "evntcre8.h"

namespace py = pybind11;

PYBIND11_MODULE(evntcre8,m) {
  m.doc() = "Geologic event creation";

  py::class_<evntcre8>(m,"evntcre8")
      .def(py::init<int,int,float,float,float>(),
           py::arg("nx"),py::arg("ny"),
           py::arg("dx"), py::arg("dy"), py::arg("dz"))
      .def("expand",[](evntcre8 &ec8,
             int itop,
             int ibot,
             int nzin,
             py::array_t<int, py::array::c_style> lyrin,
             py::array_t<float, py::array::c_style> velin,
             int nzot,
             py::array_t<int, py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot
             )
             {
               ec8.expand(itop, ibot, nzin, lyrin.mutable_data(), velin.mutable_data(),
                   nzot, lyrot.mutable_data(), velot.mutable_data());
             },
             py::arg("itop"), py::arg("ibot"),
             py::arg("nzin"), py::arg("lyrin"), py::arg("velin"),
             py::arg("nzot"), py::arg("lyrot"), py::arg("velot")
          )
      .def("deposit",[](evntcre8 &ec8,
             float vel,
             float band1,
             float band2,
             float band3,
             float layer,
             float layer_rand,
             float dev_layer,
             float dev_pos,
             int nzot,
             py::array_t<int,   py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot
             )
             {
               ec8.deposit(vel,
                   band1, band2, band3,
                   layer, layer_rand, dev_layer, dev_pos,
                   nzot, lyrot.mutable_data(), velot.mutable_data());
             },
             py::arg("vel"),
             py::arg("band1"), py::arg("band2"), py::arg("band3"),
             py::arg("layer"), py::arg("layer_rand"), py::arg("dev_layer"),
             py::arg("dev_pos"), py::arg("nzot"), py::arg("lyrot"), py::arg("velot")
      )
     .def("fault",[](evntcre8 &ec8,
             int nz,
             py::array_t<int, py::array::c_style> lyrin,
             py::array_t<float, py::array::c_style> velin,
             py::array_t<float, py::array::c_style> lblin,
             float azim,
             float begx,
             float begy,
             float begz,
             float dz,
             float daz,
             float thetashift,
             float perpdie,
             float distdie,
             float thetadie,
             float dir,
             float scalethrw,
             py::array_t<int, py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot,
             py::array_t<float, py::array::c_style> olblot,
             py::array_t<float, py::array::c_style> nlblot
             )
             {
               ec8.fault(nz,lyrin.mutable_data(),velin.mutable_data(),lblin.mutable_data(),
                   azim,begx,begy,begz,dz,daz, thetashift, perpdie, distdie, thetadie, dir, scalethrw,
                   lyrot.mutable_data(), velot.mutable_data(), olblot.mutable_data(), nlblot.mutable_data());
             },
             py::arg("nz"), py::arg("lyrin"), py::arg("velin"), py::arg("lblin"), py::arg("azim"), py::arg("begx"),
             py::arg("begy"), py::arg("begz"), py::arg("dz"), py::arg("daz"),
             py::arg("thetashift"), py::arg("perpdie"), py::arg("distdie"), py::arg("thetadie"),
             py::arg("dir"), py::arg("scalethrw"), py::arg("lyrot"), py::arg("velot"), py::arg("olblot"), py::arg("nlblot")
         )
     .def("fault_shifts2d",[](evntcre8 &ec8,
             int nz,
             py::array_t<float, py::array::c_style> lblin,
             float azim,
             float begx,
             float begz,
             float dz,
             float daz,
             float thetashift,
             float distdie,
             float thetadie,
             float scalethrw,
             py::array_t<float, py::array::c_style> olblot,
             py::array_t<float, py::array::c_style> nlblot,
             py::array_t<float, py::array::c_style> shiftx,
             py::array_t<float, py::array::c_style> shiftz
             )
             {
               ec8.fault_shifts2d(nz, lblin.mutable_data(),
                   azim, begx, begz, dz, daz, thetashift, distdie, thetadie, scalethrw,
                   olblot.mutable_data(), nlblot.mutable_data(), shiftx.mutable_data(), shiftz.mutable_data());
             },
             py::arg("nz"), py::arg("lblin"), py::arg("azim"), py::arg("begx"), py::arg("begz"),
             py::arg("dz"), py::arg("daz"), py::arg("thetashift"), py::arg("distdie"),
             py::arg("thetadie"), py::arg("scalethrw"), py::arg("olblot"), py::arg("nlblot"), py::arg("shiftx"),
             py::arg("shiftz")
         )
     .def("squish",[](evntcre8 &ec8,
             int nz,
             py::array_t<int, py::array::c_style> lyrin,
             py::array_t<float, py::array::c_style> velin,
             py::array_t<float, py::array::c_style> shftin,
             int mode,
             float azim,
             float maxshift,
             float lambda,
             float rinline,
             float rxline,
             int nzot,
             py::array_t<int, py::array::c_style> lyrot,
             py::array_t<float, py::array::c_style> velot
             )
             {
               ec8.squish(nz, lyrin.mutable_data(), velin.mutable_data(), shftin.mutable_data(),
                   mode, azim, maxshift, lambda, rinline, rxline,
                   nzot, lyrot.mutable_data(), velot.mutable_data());
             },
             py::arg("nz"), py::arg("lyrin"), py::arg("velin"), py::arg("shftin"), py::arg("mode"),
             py::arg("azim"), py::arg("maxshift"), py::arg("lambda"), py::arg("rinline"), py::arg("rxline"),
             py::arg("nzot"), py::arg("lyrot"), py::arg("velot")
         )
     .def("squish_shifts",[](evntcre8 &ec8,
             int nz,
             py::array_t<float, py::array::c_style> shftin,
             int mode,
             float azim,
             float lambda,
             float rinline,
             float rxline,
             py::array_t<float, py::array::c_style> shfty,
             py::array_t<float, py::array::c_style> shftx,
             py::array_t<float, py::array::c_style> shftz
             )
             {
               ec8.squish_shifts(nz, shftin.mutable_data(),
                   mode, azim, lambda, rinline, rxline,
                   shfty.mutable_data(), shftx.mutable_data(), shftz.mutable_data());
             },
             py::arg("nz"), py::arg("shftin"), py::arg("mode"),
             py::arg("azim"), py::arg("lambda"), py::arg("rinline"), py::arg("rxline"),
             py::arg("shfty"), py::arg("shftx"), py::arg("shftz")
         )
     .def("fill_top_bottom",[](evntcre8 &ec8,
              int nz,
              float maxshift,
              float basevel,
              py::array_t<int, py::array::c_style> lyrot,
              py::array_t<float, py::array::c_style> velot
              )
              {
                 ec8.fill_top_bottom(nz, maxshift, basevel,
                     lyrot.mutable_data(), velot.mutable_data());
              },
              py::arg("nz"),
              py::arg("maxshift"), py::arg("basevel"),
              py::arg("lyrot"), py::arg("velot")
              )
     .def("zder",[](evntcre8 &ec8,
             int nz,
             py::array_t<float, py::array::c_style> lblin,
             py::array_t<float, py::array::c_style> lblot
             )
             {
               ec8.zder(nz,lblin.mutable_data(),lblot.mutable_data());
             },
             py::arg("nz"), py::arg("lblin"), py::arg("lblot")
         )
     .def("laplacian",[](evntcre8 &ec8,
             int nz,
             py::array_t<float, py::array::c_style> lblin,
             py::array_t<float, py::array::c_style> lblot
             )
             {
               ec8.laplacian(nz,lblin.mutable_data(),lblot.mutable_data());
             },
             py::arg("nz"), py::arg("lblin"), py::arg("lblot")
         )
     .def("calcref",[](evntcre8 &ec8,
             int nz,
             py::array_t<float, py::array::c_style> vel,
             py::array_t<float, py::array::c_style> ref
             )
             {
               ec8.calcref(nz, vel.mutable_data(), ref.mutable_data());
             },
             py::arg("nz"), py::arg("vel"), py::arg("ref")
         )
     .def("calcref2d",[](evntcre8 &ec8,
					  int nx,
            int nz,
            py::array_t<float, py::array::c_style> vel,
            py::array_t<float, py::array::c_style> ref
            )
            {
               ec8.calcref2d(nx, nz, vel.mutable_data(), ref.mutable_data());
            },
						py::arg("nx"), py::arg("nz"), py::arg("vel"), py::arg("ref")
         );

}
