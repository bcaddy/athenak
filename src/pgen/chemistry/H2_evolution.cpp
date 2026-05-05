//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file H2_evolution.cpp
//! \brief Problem generator for chemistry problems using the H2 network that
//! has analytical solutions.

#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "chemistry/chemistry.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "units/units.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::H2_evolution()
//! \brief Problem Generator for the H2 tests

void ProblemGenerator::H2_evolution(ParameterInput* pin, const bool restart) {
  if (restart) return;

  // capture variables for the kernel
  auto& indcs = pmy_mesh_->mb_indcs;
  int& is = indcs.is;
  int& ie = indcs.ie;
  int& js = indcs.js;
  int& je = indcs.je;
  int& ks = indcs.ks;
  int& ke = indcs.ke;
  MeshBlockPack* pmbp = pmy_mesh_->pmb_pack;
  auto& size = pmbp->pmb->mb_size;
  auto& w0 = pmbp->phydro->w0;

  // Get the input parameters from the input file
  HydPrim1D hydro;
  const Real n_H = pin->GetReal("problem", "n_H");
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * pmbp->punit->mu() /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetReal("problem", "vx_kms");
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = 1.0;

  // H abundance at x>1
  const Real fH1 = pin->GetOrAddReal("problem", "fH1", 0.);

  // Gaussian or flat profile
  const bool gauss_profile =
      (pin->GetString("problem", "profile") == "gaussian") ? true : false;

  const Real gaussian_mean = pin->GetOrAddReal("problem", "gaussian_mean", 0.5);
  const Real gaussian_std = pin->GetOrAddReal("problem", "gaussian_std", 0.1);

  // Assign values
  const int chem_start =
      pmbp->pchemistry->get_chemistry_scalars_first_idx() - 1;
  par_for(
      "pgen_H2_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke, js,
      je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = hydro.d;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = hydro.e;

        // Assign chemistry values to this cell
        Real H_abundance, H2_abundance;
        if (gauss_profile) {
          Real& x1min = size.d_view(m).x1min;
          Real& x1max = size.d_view(m).x1max;
          int nx1 = indcs.nx1;
          const Real x = CellCenterX(i - is, nx1, x1min, x1max);

          H_abundance = n_H * Kokkos::exp(-SQR(x - gaussian_mean) /
                                          (2. * SQR(gaussian_std)));
          H2_abundance = 0.5 * (n_H - H_abundance);
        } else {
          H_abundance = fH1 * n_H;
          H2_abundance = (1. - fH1) * 0.5 * n_H;
        }

        // Write out abundances
        w0(m, chem_start + chemistry::H2Network::IH, k, j, i) = H_abundance;
        w0(m, chem_start + chemistry::H2Network::IH2, k, j, i) = H2_abundance;
      });

  // Convert primitives to conserved
  auto& u0 = pmbp->phydro->u0;
  if (pmbp->padm == nullptr) {
    pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);
  }

  return;
}
