#ifndef ODE_SOLVERS_KOKKOS_BDF_HPP_
#define ODE_SOLVERS_KOKKOS_BDF_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kokkos_bdf.hpp
//  \brief Wrapper for the Kokkos Kernels BDF ODE solver

#include <KokkosODE_BDF.hpp>
#include <string>  // NOLINT(build/include_order)

#include "athena.hpp"

namespace ode_solvers {

struct KokkosBDFSettings {
  /// Fraction of the integration interval (the hydro timestep) to use for the
  /// solver's first internal step, i.e. dt0 = first_step_frac * dt. A value of
  /// 0 (the default) gives dt0 = 0, which lets the Kokkos Kernels BDF driver
  /// auto-select its first step. This is an opt-in escape hatch: a fixed
  /// fraction is a poor global control because it is too small in the easy
  /// (near-equilibrium) regime, needlessly slowing every step, yet still too
  /// large to cure the ill-conditioned first-cycle solve for stiff networks.
  Real first_step_frac;
  /// Absolute and relative error tolerances for the BDF error test and the
  /// Newton convergence norm. Loosening them is the main control over how many
  /// internal steps a macro-step costs.
  Real atol;
  Real rtol;
};

/*!
 * \brief Kokkos Kernels' BDFSolve, with the internal step count returned.
 *
 * This mirrors KokkosODE::Experimental::BDFSolve and adds only a counter around
 * its internal while loop. Upstream keeps that loop private, so there is no way
 * to ask it how many internal steps one macro-step cost -- and that count is
 * exactly what is needed to compare chemistry cost against hydro cost, since a
 * macro-step that quietly subcycles a thousand times is not comparable to a
 * single hydro update.
 *
 * Upstream hard codes atol = 1e-6, rtol = 1e-3. Here they are arguments, so the
 * `<chemistry> kokkos_BDF_atol` and `kokkos_BDF_rtol` input keys reach the
 * solver; passing the upstream values reproduces upstream exactly.
 *
 * Re-check this against upstream whenever the pinned Kokkos Kernels version in
 * the top level CMakeLists.txt changes. Transcribed from `d7509d69` on
 * bcaddy/kokkos-kernels, where BDFStep takes max_step and returns a status, and
 * where a failed step restores y_new from y0 and stops rather than retrying.
 *
 * \param n_steps [out] The number of internal BDF steps taken.
 * \return The solver status, as upstream's BDFSolve returns.
 */
template <class ode_type, class mat_type, class vec_type, class scalar_type>
KOKKOS_FUNCTION KokkosODE::Experimental::ode_solver_status CountedBDFSolve(
    const ode_type& ode, const scalar_type t_start, const scalar_type t_end,
    const scalar_type initial_step, const scalar_type max_step,
    const scalar_type atol, const scalar_type rtol, const vec_type& y0,
    const vec_type& y_new, mat_type& temp, mat_type& temp2, int& n_steps) {
  using KAT = Kokkos::ArithTraits<scalar_type>;
  using ode_solver_status = KokkosODE::Experimental::ode_solver_status;

  auto rhs = Kokkos::subview(temp, Kokkos::ALL(), 0);
  auto update = Kokkos::subview(temp, Kokkos::ALL(), 1);

  int order = 1, num_equal_steps = 0;
  constexpr scalar_type min_factor = 0.2;
  scalar_type dt = initial_step;
  scalar_type t = t_start;

  constexpr int max_newton_iters = 10;

  // Compute rhs = f(t_start, y0)
  ode.evaluate_function(t_start, 0, y0, rhs);

  // Check if we need to compute the initial time step size.
  if (initial_step == KAT::zero()) {
    KokkosODE::Impl::initial_step_size(ode, order, t_start, atol, rtol, y0, rhs,
                                       temp, dt);
  }

  // Initialize D(:, 0) = y0 and D(:, 1) = dt*rhs
  auto D = Kokkos::subview(temp, Kokkos::ALL(), Kokkos::pair<int, int>(2, 10));
  for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
    D(eqIdx, 0) = y0(eqIdx);
    D(eqIdx, 1) = dt * rhs(eqIdx);
    rhs(eqIdx) = 0;
  }

  n_steps = 0;
  ode_solver_status status = ode_solver_status::SUCCESS;
  while (t < t_end) {
    status = KokkosODE::Impl::BDFStep(
        ode, t, dt, t_end, max_step, order, num_equal_steps, max_newton_iters,
        atol, rtol, min_factor, y0, y_new, rhs, update, temp, temp2);

    if (status != ode_solver_status::SUCCESS) {
      for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
        y_new(eqIdx) = y0(eqIdx);
      }
      break;
    }

    for (int eqIdx = 0; eqIdx < ode.neqs; ++eqIdx) {
      y0(eqIdx) = y_new(eqIdx);
    }
    ++n_steps;
  }
  return status;
}

/*!
 * \brief Solve a system of ODEs using the BDF solver from Kokkos Kernels
 *
 * \tparam T The type of the ODE system to solve
 */
template <typename ode_t>
class KokkosBDF {
 public:
  // ----- Constructor & Destructor -----
  KOKKOS_FUNCTION
  KokkosBDF(KokkosBDFSettings const settings, ode_t& ode_system,
            Real const t_start, Real const dt)
      : ode_system(ode_system),
        t_start(t_start),
        dt(dt),
        t_end(t_start + dt),
        dt0(settings.first_step_frac * dt),
        max_step(dt),
        atol(settings.atol),
        rtol(settings.rtol),
        temp_(&temp_buffer_[0][0], ode_t::neqs, 23 + 2 * ode_t::neqs + 4),
        temp2_(&temp2_buffer_[0][0], 6, 7) {}
  KOKKOS_FUNCTION
  ~KokkosBDF() = default;

  // ----- Variables -----
  /// The system of ODEs to solve
  ode_t& ode_system;
  /// The starting time for this solve
  const Real t_start;
  /// The amount of time to evolve the system of equations
  const Real dt;
  /// Time to integrate to
  const Real t_end;
  /// First time step size, if zero then the solver will decide
  const Real dt0;
  /// The maximum internal time step; zero lets the solver decide. The
  /// constructor sets it to the hydro step. Honoured by the pinned Kokkos
  /// Kernels; earlier versions discarded it (`(void)max_step;`).
  const Real max_step;
  /// Error tolerances passed to the BDF error test and the Newton norm.
  const Real atol;
  const Real rtol;
  /// Number of internal BDF steps the last SolveODE() call took. Diagnostic
  /// only: per-cell chemistry cost scales with this.
  int n_substeps = 0;

  /*!
   * \brief Get the settings for the  ODE solver from the input file
   *
   * \param pin The ParameterInput object
   * \param module The physics module that this ODE solver is called in. The
   * name should match the block name in the input file for the physics module.
   * \return KokkosBDFSettings The settings for the Kokkos BDF solver
   */
  static KokkosBDFSettings GetSettings(ParameterInput* pin,
                                       std::string module) {
    // Default 0 => dt0 = 0 => the solver auto-selects its first step. A fixed
    // fraction of the macro-step is a poor global control (too small in the
    // easy regime, too large in the stiff first cycle), so it is opt-in only.
    // The defaults are the values Kokkos Kernels hard codes, so an input file
    // that sets neither key reproduces upstream exactly.
    return KokkosBDFSettings{
        pin->GetOrAddReal(module, "kokkos_BDF_first_step_frac", 0.0),
        pin->GetOrAddReal(module, "kokkos_BDF_atol", 1.0e-6),
        pin->GetOrAddReal(module, "kokkos_BDF_rtol", 1.0e-3)};
  }

  KOKKOS_FUNCTION
  void SolveODE() {
    auto const status =
        CountedBDFSolve(ode_system, t_start, t_end, dt0, max_step, atol, rtol,
                        ode_system.y, ode_system.y_new, temp_, temp2_,
                        n_substeps);

    // Note that this may not trigger an MPI_Abort, instead just aborting a
    // single rank. If that becomes a problem it can be replaced with a failure
    // flag that is checked on the host.
    if (status != KokkosODE::Experimental::ode_solver_status::SUCCESS) {
      // A non-success status means the solver could not complete the interval
      // and y holds the solution at some time < t_end
      Kokkos::printf(
          "KokkosBDF: BDF solve failed with status %d at t_start=%.17g "
          "dt=%.17g\n",
          static_cast<int>(status), t_start, dt);
      Kokkos::abort("KokkosBDF: BDF ODE solve failed");
    }
  }

 private:
  // temporary storage for inside the BDF solver
  Real temp_buffer_[ode_t::neqs][23 + 2 * ode_t::neqs + 4];
  Real temp2_buffer_[6][7];
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp_;
  Kokkos::View<Real**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged> temp2_;
};
}  // namespace ode_solvers
#endif  // ODE_SOLVERS_KOKKOS_BDF_HPP_
