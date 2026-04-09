//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file forward_euler.cpp
//! \brief implementation of ForwardEuler class

#include "chemistry/ode_solver/forward_euler.hpp"

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/chemistry.hpp"
#include "parameter_input.hpp"

namespace chemistry {
ForwardEuler::ForwardEuler(MeshBlockPack* ppack, ParameterInput* pin)
    : ODEBase(ppack, pin),
      cfl_cool_sub(pin->GetOrAddReal("chemistry", "cfl_cool_sub", 0.1)),
      abundance_floor(pin->GetOrAddReal("chemistry", "abundance_floor", 1e-3)),
      nsub_max(pin->GetOrAddInteger("chemistry", "nsub_max", 1e5)),
      n_species(ppack->pchemistry->nscalars_chemistry) {};

void ForwardEuler::Integrate(const Real tinit, const Real dt) {
  std::cout << "placeholder text" << std::endl;
}

}  // namespace chemistry