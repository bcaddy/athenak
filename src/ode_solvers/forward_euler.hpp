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
template <typename T>
class ForwardEuler {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION ForwardEuler(T& ode_system, Real const t_start, Real const dt)
      : ode_system(ode_system), t_start(t_start), dt(dt) {}
  KOKKOS_FUNCTION ~ForwardEuler() = default;

  // ----- Variables -----
  /// A small number approximately equal to
  /// 1024*std::numeric_limits<float>::min()
  static constexpr Real small = 1e-35;
  /// The floor for chemical abundances
  const Real yfloor = 1.0e-3;
  /// The CFL number for the forward euler subcycling. Lowering this has no
  /// impact on the solution for the H2 network
  static constexpr Real cfl_cool_subcycle = 0.1;
  /// The maximum number of forward euler iterations
  static constexpr Real max_iterations = 1e5;
  /// The system of ODEs to solve
  T& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// If the solver failed to converge within the allocated number of cycles
  bool failed = false;

  KOKKOS_FUNCTION
  void SolveODE() {
    // ------ Solve the ODEs ------
    int icount = 0;
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
        dt_subcycle = cfl_cool_subcycle * dt_subcycle;

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

      // check if convergence is established within max_iterations.  If not,
      // trigger a failure
      if (icount > max_iterations) {
        failed = true;
      }
    }
  }
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_FORWARD_EULER_HPP_
