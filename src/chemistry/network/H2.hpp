#ifndef CHEMISTRY_NETWORK_H2_HPP_
#define CHEMISTRY_NETWORK_H2_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2.hpp
//  \brief The implementation for the struct for the H2 chemistry network

#include "athena.hpp"
#include "chemistry/thermo/thermo.hpp"

namespace chemistry {
struct H2Network {
  H2Network(Real const units_time_cgs, Real const units_energy_density_cgs)
      : units_time_cgs(units_time_cgs),
        units_energy_density_cgs(units_energy_density_cgs) {}

  // Number of equations
  static constexpr int neqs = 3;

  // Species indices within the ODE system
  enum {
    IIE = 0,  // internal energy
    IH2 = 1,  // H_2
    IH = 2    // H
  };

  // ----- cell values -----
  Real n_H;  // The number density of hydrogen
  Real internal_energy;

  // ----- unit conversion factors -----
  Real const units_time_cgs;
  Real const units_energy_density_cgs;

  // ----- Reaction rate constants -----
  static constexpr Real k_gr = 3.0e-17;
  // xi_cr is the primary cosmic-ray ionization rate per H
  static constexpr Real xi_cr = 2.0e-16;
  static constexpr Real k_cr = 3.0 * xi_cr;

  // ----- Member Functions -----
  template <class vec_type1, class vec_type2>
  KOKKOS_FUNCTION void evaluate_function(const double /*t*/,
                                         const double /*dt*/,
                                         const vec_type1& y,
                                         const vec_type2& f) {
    // ----- Internal energy equation -----
    static constexpr Real x_He = 0.1;
    static constexpr Real x_e = 0.0;
    const Real x_H2 = y[IH2];

    static constexpr Real T_floor = 1.;  // temperature floor for cooling
    // energy per hydrogen atom
    const Real E_ergs = internal_energy * units_energy_density_cgs / n_H;
    const Real T = E_ergs / Thermo::CvCold(x_H2, x_He, x_e);
    if (T < T_floor) {
      f(IIE) = 0;
    } else {
      const Real dEdt = -Thermo::alpha_GD_ * n_H * std::sqrt(T) * T;
      // convert to code units
      f(IIE) = (dEdt * n_H / units_energy_density_cgs) * units_time_cgs;
    }

    // ----- Abundance equations -----
    // cr = cosmic ray, gr = dust grain
    const Real rate_cr = k_cr * y[IH2];
    const Real rate_gr = k_gr * n_H * y[IH];

    // H_2 equation
    f(IH2) = rate_gr - rate_cr;
    // H equation
    f(IH) = 2 * (rate_cr - rate_gr);

    // Convert abundances back to code units
    for (size_t i = 1; i < neqs; i++) {
      f(i) *= units_time_cgs;
    }

    // original equations
    // f(0) = -0.04 * y(0) + 1.e4 * y(1) * y(2);
    // f(1) = 0.04 * y(0) - 1.e4 * y(1) * y(2) - 3.e7 * y(1) * y(1);
    // f(2) = 3.e7 * y(1) * y(1);
  }

  // template <class vec_type, class mat_type>
  // KOKKOS_FUNCTION void evaluate_jacobian(const double /*t*/,
  //                                        const double /*dt*/, const vec_type&
  //                                        y, const mat_type& jac) {
  //   jac(0, 0) = -0.04;
  //   jac(0, 1) = 1.e4 * y(2);
  //   jac(0, 2) = 1.e4 * y(1);
  //   jac(1, 0) = 0.04;
  //   jac(1, 1) = -1.e4 * y(2) - 3.e7 * 2.0 * y(1);
  //   jac(1, 2) = -1.e4 * y(1);
  //   jac(2, 0) = 0.0;
  //   jac(2, 1) = 3.e7 * 2.0 * y(1);
  //   jac(2, 2) = 0.0;
  // }
};
}  // namespace chemistry
#endif  // CHEMISTRY_NETWORK_H2_HPP_
