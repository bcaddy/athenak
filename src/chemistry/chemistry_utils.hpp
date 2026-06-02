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
KOKKOS_INLINE_FUNCTION int LinearInterpIndex(const int len, const Real xarr[],
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
KOKKOS_INLINE_FUNCTION Real LP1Di(const Real* xarr, const Real* data,
                                  const int ix, const Real x) {
  return LinearInterp(xarr[ix], xarr[ix + 1], data[ix], data[ix + 1], x);
}

//---------------------------------------------------------------------------------------
/*!
 * \brief 2D array bi-linear interpolation with index provided
 */
KOKKOS_INLINE_FUNCTION Real LP2Di(const Real* xarr, const Real* yarr,
                                  const int lenx, const int ix, const int iy,
                                  const Real* data, const Real x,
                                  const Real y) {
  Real fl1, fl2;
  const Real x0 = xarr[ix];
  const Real x1 = xarr[ix + 1];
  fl1 = LinearInterp(x0, x1, data[iy * lenx + ix], data[iy * lenx + ix + 1], x);
  fl2 = LinearInterp(x0, x1, data[(iy + 1) * lenx + ix],
                     data[(iy + 1) * lenx + ix + 1], x);
  return LinearInterp(yarr[iy], yarr[iy + 1], fl1, fl2, y);
}

}  // namespace interpolation

}  // namespace chemistry

#endif  // CHEMISTRY_CHEMISTRY_UTILS_HPP_
