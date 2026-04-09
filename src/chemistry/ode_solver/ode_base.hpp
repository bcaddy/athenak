#ifndef CHEMISTRY_ODE_BASE_HPP_
#define CHEMISTRY_ODE_BASE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ode_base.hpp
//  \brief definitions for ODE class

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "parameter_input.hpp"

namespace chemistry {
//! \class ODEBase
//! \brief Base class for the ODE solver(s)
class ODEBase {
 public:
  ODEBase(MeshBlockPack* ppack, ParameterInput* pin)
      : pmy_pack(ppack),
        output_zones_per_sec(pin->GetOrAddBoolean(
            "chemistry", "output_zones_per_sec", false)) {};
  virtual ~ODEBase() = default;
  virtual void Integrate(const Real tinit, const Real dt) = 0;

 private:
  MeshBlockPack* const pmy_pack;
  bool output_zones_per_sec;  // option to output solver performance
};

}  // namespace chemistry
#endif  // CHEMISTRY_ODE_BASE_HPP_
