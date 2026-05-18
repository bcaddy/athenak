#ifndef ODE_SOLVERS_FORWARD_EULER_HPP_
#define ODE_SOLVERS_FORWARD_EULER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file forward_euler.hpp
//  \brief The implementation of a forward euler solver for solving systems of
//  ODEs

#include "athena.hpp"

namespace ode_solvers {

struct FESettings {
  /// The max number of subcycles the Forwared Euler solver should execute
  unsigned int fe_n_subcycle_max;
  /// CFL number for the subcycles
  Real fe_cfl;
};

template <typename T>
class ForwardEuler {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION ForwardEuler(FESettings const settings, T& ode_system,
                               Real const t_start, Real const dt)
      : ode_system(ode_system),
        fe_cfl(settings.fe_cfl),
        fe_n_subcycle_max(settings.fe_n_subcycle_max),
        t_start(t_start),
        dt(dt) {}
  KOKKOS_FUNCTION ~ForwardEuler() = default;

  // ----- Variables -----
  /// A small number approximately equal to
  /// 1024*std::numeric_limits<float>::min()
  static constexpr Real small = 1e-35;
  /// The floor for chemical abundances
  const Real yfloor = 1.0e-3;
  /// The CFL number for the forward euler subcycling. Lowering this has no
  /// impact on the solution for the H2 network
  const Real fe_cfl;
  /// The maximum number of forward euler iterations
  unsigned int fe_n_subcycle_max ;
  /// The system of ODEs to solve
  T& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// If the solver failed to converge within the allocated number of cycles
  bool failed = false;

  /*!
   * \brief Get the settings for the Forward Euler ODE solver from the input
   * file
   *
   * \param pin The ParameterInput object
   * \return FESettings The settings for the Forward Euler solver
   */
  static FESettings GetSettings(ParameterInput* pin) {
    unsigned int fe_n_subcycle_max =
        pin->GetOrAddInteger("chemistry", "fe_n_subcycle_max", 1e5);
    Real fe_cfl = pin->GetOrAddReal("chemistry", "fe_cfl", 0.1);

    return FESettings{fe_cfl};//{fe_n_subcycle_max, fe_cfl};
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    // ------ Solve the ODEs ------
    unsigned int icount = 0;
    Real t_now = t_start;
    Real t_end = t_start + dt;
    while (t_now < t_end) {
      // Evaluate the ODEs
      ode_system.evaluate_function();

      // Compute the chemistry time scale
      Real dt_subcycle;
      {
        // calculate chemistry timescale
        dt_subcycle = Kokkos::reduction_identity<Real>::min();
        for (int s_idx = 0; s_idx < ode_system.neqs; s_idx++) {
          // put floor in species abundance
          Real const yf = Kokkos::max(ode_system.y(s_idx), yfloor);

          // Compute the value to reduce
          // NOLINTNEXTLINE(build/include_what_you_use)
          dt_subcycle = Kokkos::min(
              dt_subcycle, Kokkos::abs(yf / (ode_system.f(s_idx) + small)));
        }
        dt_subcycle = fe_cfl * dt_subcycle;

        // If t_now + dt_subcycle is greater than t_end then lower the
        // timestep accordingly
        // NOLINTNEXTLINE(build/include_what_you_use)
        dt_subcycle = Kokkos::min(dt_subcycle, t_end - t_now);
      }

      // Advance one subcycle
      {
        for (int s_idx = 0; s_idx < ode_system.neqs; s_idx++) {
          ode_system.y(s_idx) += ode_system.f(s_idx) * dt_subcycle;
        }
      }

      // Update timing
      t_now += dt_subcycle;
      icount++;

      // check if convergence is established within fe_n_subcycle_max.  If not,
      // trigger a failure
      if (icount > fe_n_subcycle_max) {
        failed = true;
      }
    }
  }
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_FORWARD_EULER_HPP_
