#ifndef CHEMISTRY_CHEMISTRY_UTILS_HPP_
#define CHEMISTRY_CHEMISTRY_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chemistry_utils.hpp
//  \brief utilities for chemistry

#include <Kokkos_ArithTraits.hpp>

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

namespace interpolation {
// ---------------------------------------------------------------------------------------
/*!
 * \brief Find the proper index for interpolation
 *
 * \param len The size of xarr
 * \param xarr The array to find the interpolation index in
 * \param x The value to interpolate around
 * \return int The index of the xarr with the first instance where x > xarr[i]
 */
KOKKOS_INLINE_FUNCTION size_t LinearInterpIndex(const size_t len,
                                                const Real xarr[],
                                                const Real x) {
  if (x < xarr[0]) {
    return 0;
  } else if (x > xarr[len - 1]) {
    return len - 2;
  } else {
    int i = 0;
    while (x > xarr[i]) {
      i++;
    }
    return i - 1;
  }
}

template <std::size_t N>
KOKKOS_INLINE_FUNCTION size_t LinearInterpIndex(
    const size_t len, const Kokkos::Array<Real, N> xarr, const Real x) {
  if (x < xarr[0]) {
    return 0;
  } else if (x > xarr[len - 1]) {
    return len - 2;
  } else {
    int i = 0;
    while (x > xarr[i]) {
      i++;
    }
    return i - 1;
  }
}

//---------------------------------------------------------------------------------------
KOKKOS_INLINE_FUNCTION Real LinearInterp(const Real x0, const Real x1,
                                         const Real y0, const Real y1,
                                         const Real x) {
  return y0 + ((y1 - y0) / (x1 - x0)) * (x - x0);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief Interpolation with index provided.
 */
template <std::size_t N_xarr, std::size_t N_data>
KOKKOS_INLINE_FUNCTION Real LP1Di(const Kokkos::Array<Real, N_xarr> xarr,
                                  const Kokkos::Array<Real, N_data> data,
                                  const int ix, const Real x) {
  return LinearInterp(xarr[ix], xarr[ix + 1], data[ix], data[ix + 1], x);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief 2D array bi-linear interpolation with index provided
 */
template <std::size_t N_xarr, std::size_t N_yarr, std::size_t N_data>
KOKKOS_INLINE_FUNCTION Real LP2Di(const Kokkos::Array<Real, N_xarr> xarr,
                                  const Kokkos::Array<Real, N_yarr> yarr,
                                  const int lenx, const int ix, const int iy,
                                  const Kokkos::Array<Real, N_data> data,
                                  const Real x, const Real y) {
  Real fl1, fl2;
  const Real x0 = xarr[ix];
  const Real x1 = xarr[ix + 1];
  fl1 = LinearInterp(x0, x1, data[iy * lenx + ix], data[iy * lenx + ix + 1], x);
  fl2 = LinearInterp(x0, x1, data[(iy + 1) * lenx + ix],
                     data[(iy + 1) * lenx + ix + 1], x);
  return LinearInterp(yarr[iy], yarr[iy + 1], fl1, fl2, y);
}

}  // namespace interpolation

namespace rates {
//----------------------------------------------------------------------------------------
/*!
 * \brief Rate coefficient for CII recombination, radiative plus dielectronic,
 * from Badnell (2003, 2006). Shared by the GOW17 and NCR networks.
 *
 * \param T The temperature in K
 * \return Real The rate coefficient for C+ + e -> C in cm^3 s^-1
 */
KOKKOS_INLINE_FUNCTION Real CIIRecombination(const Real T) {
  constexpr Real A = 2.995e-9;
  constexpr Real B = 0.7849;
  constexpr Real T0 = 6.670e-3;
  constexpr Real T1 = 1.943e6;
  constexpr Real C = 0.1597;
  constexpr Real T2 = 4.955e4;
  const Real BN = B + C * Kokkos::exp(-T2 / T);
  const Real term1 = Kokkos::sqrt(T / T0);
  const Real term2 = Kokkos::sqrt(T / T1);
  const Real alpha_rr = A / (term1 * Kokkos::pow(1.0 + term1, 1.0 - BN) *
                             Kokkos::pow(1.0 + term2, 1.0 + BN));
  // T^(-3/2) = T0^(-3/2) * (T/T0)^(-3/2) = T0^(-3/2) / term1^3, so the
  // power comes for free from term1 instead of a pow().
  constexpr Real T0_inv15_ = 1.83574032975375144e3;
  const Real alpha_dr = (T0_inv15_ / (term1 * term1 * term1)) *
                        (6.346e-9 * Kokkos::exp(-1.217e1 / T) +
                         9.793e-09 * Kokkos::exp(-7.38e1 / T) +
                         1.634e-06 * Kokkos::exp(-1.523e+04 / T));
  return (alpha_rr + alpha_dr);
}

//----------------------------------------------------------------------------------------
/*!
 * \brief Rate coefficient for H2 formation on dust grains with the
 * temperature dependence of Hollenbach & McKee (1979), taking Tgr2 = 0 and
 * renormalized to kgr0 = 3e-17 near 200 > T > 50 K. TIGRESS-NCR (Kim+23)
 * implementation. Shared by the GOW17 and NCR networks.
 *
 * \param T The temperature in K
 * \return Real The H2 formation rate coefficient on grains in cm^3 s^-1, per
 * unit dust metallicity
 */
KOKKOS_INLINE_FUNCTION Real H2FormationGrainHM79(const Real T) {
  const Real kgr0 = 3.0e-17;
  const Real T2 = T * 1e-2;
  return kgr0 * Kokkos::sqrt(T2) * 2.0 /
         (1 + 0.4 * Kokkos::sqrt(T2) + 0.2 * T2 + 0.08 * T2 * T2);
}

//----------------------------------------------------------------------------------------
/*!
 * \brief The temperature dependence of case B radiative recombination of H+,
 * Ferland et al. (1992): (315614/T)^1.5 (1 + (115188/T)^0.407)^-2.242.
 * Shared by the GOW17 and NCR networks.
 *
 * \details 315614 K = 2*13.6 eV/k_B. pow(c/T, e) is computed as
 * exp(e*(ln c - ln T)) to reuse lnT. The outer power has a non-constant base,
 * so it stays a pow.
 *
 * \param lnT ln(T), T the temperature in K
 * \return Real The factor, which times 2.753e-14 cm^3 s^-1 is the rate
 * coefficient
 */
KOKKOS_INLINE_FUNCTION Real CaseBRecombHII(const Real lnT) {
  constexpr Real ln315614_ = 1.26622752269192862e1;
  constexpr Real ln115188_ = 1.16543208551515374e1;
  return Kokkos::exp(1.5 * (ln315614_ - lnT)) *
         Kokkos::pow(1.0 + Kokkos::exp(0.407 * (ln115188_ - lnT)), -2.242);
}

//----------------------------------------------------------------------------------------
/*!
 * \brief Rate coefficient for collisional ionization of HI by electrons,
 * Janev et al. (1987). Shared by the GOW17 and NCR networks, which apply
 * their own temperature cutoffs.
 *
 * \param lnTe ln(T in eV), i.e. ln(8.6173e-5 T) with T in K
 * \return Real The rate coefficient in cm^3 s^-1
 */
KOKKOS_INLINE_FUNCTION Real HICollIonJanev87(const Real lnTe) {
  return Kokkos::exp(
      -3.271396786e1 +
      (1.35365560e1 +
       (-5.73932875 +
        (1.56315498 +
         (-2.877056e-1 +
          (3.48255977e-2 +
           (-2.63197617e-3 +
            (1.11954395e-4 + (-2.03914985e-6) * lnTe) * lnTe) *
               lnTe) *
              lnTe) *
             lnTe) *
            lnTe) *
           lnTe) *
          lnTe);
}

//----------------------------------------------------------------------------------------
/*!
 * \brief Rate coefficient for grain-assisted recombination, the fit of
 * Weingartner & Draine (2001) eq. (8), per unit dust metallicity. pow(x, e)
 * is computed as exp(e*ln x) so that the species share lnT and lnPsi.
 *
 * \param c The 7 fit coefficients C_0 to C_6 of the ion
 * \param lnT ln(T), T the temperature in K
 * \param lnPsi ln(psi), psi = G sqrt(T) / n_e the grain charging parameter
 * \return Real The rate coefficient in cm^3 s^-1
 */
KOKKOS_INLINE_FUNCTION Real GrainRecombWD01(const Real (&c)[7], const Real lnT,
                                            const Real lnPsi) {
  return 1.0e-14 * c[0] /
         (1.0 + c[1] * Kokkos::exp(c[2] * lnPsi) *
                    (1.0 + c[3] * Kokkos::exp(c[4] * lnT) *
                               Kokkos::exp((-c[5] - c[6] * lnT) * lnPsi)));
}

//----------------------------------------------------------------------------------------
/// The coefficients of H2 collisional dissociation, see H2CollDissGM07
struct H2CollDissCoeffs {
  /// ln of the low and high density limits of the rate coefficients for
  /// H2 + H -> 3 H (k9) and H2 + H2 -> H2 + 2 H (k10), in cm^3 s^-1
  Real ln_k9l, ln_k9h, ln_k10l, ln_k10h;
  /// The critical densities of HI and H2 in cm^-3
  Real ncrHI, ncrH2;
};

/*!
 * \brief The density dependent collisional dissociation of H2 by HI and H2,
 * Glover & Mac Low (2007): the low and high density limits of the rate
 * coefficients and the critical densities. The networks combine these with
 * their own critical density mixing and n/n_cr weights. Shared by the GOW17
 * and NCR networks.
 *
 * \details The rate coefficients are returned as natural logs since the
 * networks interpolate them in log space, and pow(c, e) is written
 * exp(e*ln c) to reuse lnT.
 *
 * \param T The temperature in K
 * \param lnT ln(T)
 * \param logT4 log10(T/1e4)
 * \return H2CollDissCoeffs The coefficients
 */
KOKKOS_INLINE_FUNCTION H2CollDissCoeffs H2CollDissGM07(const Real T,
                                                       const Real lnT,
                                                       const Real logT4) {
  constexpr Real ln10 = 2.30258509299404590e0;
  constexpr Real ln6_67em12 = -2.57334012560010166e1;
  constexpr Real ln3_52em9 = -1.94648048473364064e1;
  constexpr Real ln5_996em30 = -6.72864602095810227e1;
  constexpr Real ln1_3em9 = -2.04609015724789209e1;
  const Real Tinv = 1.0 / T;
  H2CollDissCoeffs c;
  // 6.67e-12 sqrt(T) exp(-(1 + 63590/T))
  c.ln_k9l = ln6_67em12 + 0.5 * lnT - (1. + 63590. * Tinv);
  // 3.52e-9 exp(-43900/T)
  c.ln_k9h = ln3_52em9 - 43900.0 * Tinv;
  // 5.996e-30 T^4.1881 (1 + 6.761e-6 T)^-5.6881 exp(-54657.4/T)
  c.ln_k10l = ln5_996em30 + 4.1881 * lnT -
              5.6881 * Kokkos::log(1.0 + 6.761e-6 * T) - 54657.4 * Tinv;
  // 1.3e-9 exp(-53300/T)
  c.ln_k10h = ln1_3em9 - 53300.0 * Tinv;
  // pow(10, x) == exp(x*ln10), which avoids the general power routine
  c.ncrHI =
      Kokkos::exp(ln10 * (3.0 - 0.416 * logT4 - 0.327 * logT4 * logT4));
  c.ncrH2 =
      Kokkos::exp(ln10 * (4.845 - 1.3 * logT4 + 1.62 * logT4 * logT4));
  return c;
}

/// GrainRecombWD01 for H+. Shared by the GOW17 and NCR networks
KOKKOS_INLINE_FUNCTION Real GrainRecombHplus(const Real lnT, const Real lnPsi) {
  constexpr Real cHp[7] = {12.25,    8.074e-6, 1.378,   5.087e2,
                           1.586e-2, 0.4723,   1.102e-5};
  return GrainRecombWD01(cHp, lnT, lnPsi);
}

/// GrainRecombWD01 for C+. Shared by the GOW17 and NCR networks
KOKKOS_INLINE_FUNCTION Real GrainRecombCplus(const Real lnT, const Real lnPsi) {
  constexpr Real cCp[7] = {45.58,    6.089e-3, 1.128,   4.331e2,
                           4.845e-2, 0.8120,   1.333e-4};
  return GrainRecombWD01(cCp, lnT, lnPsi);
}
}  // namespace rates

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
  RegisterArray<Real, network_t::neqs> f0, fp;

  // Evaluate the unperturbed f0
  network.evaluate_function(t, dt, y_in, f0);

  // The perturbation to add to each element in turn
  const Real perturbation_factor =
      Kokkos::sqrt(Kokkos::ArithTraits<Real>::epsilon());

  for (int j = 0; j < network.neqs; ++j) {
    // Add the perturbation to the jth element
    const Real perturbation =
        perturbation_factor * Kokkos::fmax(Kokkos::abs(y_in(j)), Real(1.0e-6));
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
