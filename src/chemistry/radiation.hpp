#ifndef CHEMISTRY_RADIATION_HPP_
#define CHEMISTRY_RADIATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation.hpp
//  \brief Interfaces to radiative transfer for the chemistry code

#include "athena.hpp"
#include "bvals/bvals.hpp"

namespace chemistry {
class Radiation {
 public:
  Radiation(MeshBlockPack* ppack, ParameterInput* pin) {
    // Get parameters
    int nfreq = pin->GetOrAddInteger("chem_radiation", "n_frequency", 1);
    const Real G0 = pin->GetReal("chem_radiation", "G0");
    const Real cr_rate = pin->GetOrAddReal("chem_radiation", "CR", 2e-16);

    // Assign values in host array
    HostArray1D<Real> ir_host("chemistry_ir_host", nfreq);
    for (size_t i = 0; i < ir_host.size(); i++) {
      ir_host(i) = G0;
    }
    ir_host(ir_host.size() - 1) += cr_rate;

    // Copy to device
    Kokkos::realloc(ir, ir_host.size());
    Kokkos::deep_copy(ir, ir_host);
  }
  ~Radiation() {}

  DvceArray1D<Real> ir;

 private:
};

}  // namespace chemistry
#endif  // CHEMISTRY_RADIATION_HPP_
