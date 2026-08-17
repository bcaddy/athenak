#ifndef CHEMISTRY_CHEMISTRY_UTILS_HPP_
#define CHEMISTRY_CHEMISTRY_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_utils.hpp
//  \brief utilities for chemistry

#include "athena.hpp"
#include "utils/register_array.hpp"

namespace chemistry {

/*!
 * \brief A struct to hold the creation and destruction rates
 *
 * \tparam N The size of each array
 */
template <std::size_t N>
struct CDRates_t {
  RegisterArray<Real, N> creation, destruction;
};

/*!
 * \brief Compute the numerical Jacobian for a chemical network.
 *
 * \tparam network_t The network object
 * \tparam vec_type The type of y_in
 * \tparam mat_type The type of jac
 * \param[in] network The chemical network to compute the Jacobian of
 * \param[in] t Current time
 * \param[in] dt Time step
 * \param[in] y_in The current state
 * \param[out] jac The Jacobian
 */
template <class network_t, class vec_type, class mat_type>
KOKKOS_FUNCTION void numerical_jacobian(const network_t& network, const Real t,
                                        const Real dt, const vec_type& y_in,
                                        const mat_type& jac) {
  RegisterArray<Real, network.neqs> f0, fp;

  // Evaluate the unperturbed f0
  network.evaluate_function(t, dt, y_in, f0);

  // The perturbation to add to each element in turn
  const Real perturbation_factor =
      Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon());

  for (int j = 0; j < network.neqs; ++j) {
    // Add the perturbation to the jth element
    const Real perturbation =
        perturbation_factor * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0));
    const Real y_unperturbed = y_in(j);
    y_in(j) += perturbation;

    // Compute the perturbed values of fp
    network.evaluate_function(t, dt, y_in, fp);

    // realized step, robust to rounding
    const Real inverse_diff = Real(1.0) / perturbation;

    // Update the Jacobian
    for (int k = 0; k < network.neqs; ++k) {
      jac(k, j) = (fp(k) - f0(k)) * inverse_diff;
    }

    // Reset the perturbed field
    y_in(j) = y_unperturbed;
  }
}
}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_UTILS_HPP_
