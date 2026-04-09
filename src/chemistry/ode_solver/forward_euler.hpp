#ifndef CHEMISTRY_ODE_FORWARD_EULER_HPP_
#define CHEMISTRY_ODE_FORWARD_EULER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file forward_euler.hpp
//  \brief definitions for ForwardEuler class

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "chemistry/ode_solver/ode_base.hpp"
#include "parameter_input.hpp"

namespace chemistry {
//! \class ForwardEuler
//! \brief Forward Euler ODE solver
class ForwardEuler : public ODEBase {
 public:
  ForwardEuler(MeshBlockPack* ppack, ParameterInput* pin);
  ~ForwardEuler() = default;
  void Integrate(const Real tinit, const Real dt);

 private:
  // cfl number for subcycling
  Real const cfl_cool_sub;
  // species abundance floor for calculating the cooling time
  Real const abundance_floor;
  // maximum number of substeps
  int const nsub_max;
  // The number of species
  int const n_species;
};

}  // namespace chemistry
#endif  // CHEMISTRY_ODE_FORWARD_EULER_HPP_
