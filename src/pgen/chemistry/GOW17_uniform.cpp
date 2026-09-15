//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file GOW17_uniform.cpp
//! \brief Problem generator for chemistry problem with a uniform state using
//! the GOW17 network

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
//! \fn ProblemGenerator::GOW17_uniform()
//! \brief Problem Generator for the GOW17 test problem with a uniform state

void ProblemGenerator::GOW17Uniform(ParameterInput* pin, const bool restart) {
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
  auto& w0 = pmbp->phydro->w0;
  auto& u0 = pmbp->phydro->u0;

  // Chemistry is optional here. With no <chemistry> block the module is never
  // constructed, and this problem generator sets up the same uniform state as a
  // pure hydro run, which is what separates chemistry cost from hydro cost in a
  // performance comparison.
  const bool chemistry_on = (pmbp->pchemistry != nullptr);

  // ----- Get the input parameters from the input file -----
  // Hydro values
  const Real n_H = pin->GetReal("problem", "n_H");
  const Real iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  HydPrim1D hydro;
  // Matches the <chemistry> mu_H default so a hydro-only run has the same density.
  const Real mu_H = chemistry_on ? pmbp->pchemistry->mu_H
                                 : pin->GetOrAddReal("problem", "mu_H", 1.4);
  hydro.d = n_H * pmbp->punit->hydrogen_mass_cgs * mu_H /
            pmbp->punit->density_cgs();
  hydro.vx = pin->GetOrAddReal("problem", "vx_kms", 0.0);
  hydro.vy = 0.0;
  hydro.vz = 0.0;
  hydro.e = n_H * SQR(iso_cs) / (pmbp->phydro->peos->eos_data.gamma - 1.0);

  // Chemistry values. The array is always allocated (its length is a compile-time
  // constant) but is only filled and read when the chemistry module is present.
  DualArray1D<Real> initial_chemistry("initial_chemistry",
                                      chemistry::GOW17Network::neqs - 1);
  if (chemistry_on) {
    const Real init_default = pin->GetOrAddReal("problem", "init_default", 0.0);
    for (size_t i = 0; i < chemistry::GOW17Network::neqs - 1; i++) {
      // Determine the name in the parameter file
      const auto name = chemistry::GOW17Network::species_names[i];
      const auto init_name = std::string("init_") + std::string(name);

      // Get the value and save it
      const Real val = pin->GetOrAddReal("problem", init_name, init_default);
      initial_chemistry.view_host()(i) = val;
    }
  }

  // Copy intializing data to the device
  initial_chemistry.modify_host();
  initial_chemistry.sync_device();
  auto initial_chemistry_d = initial_chemistry.view_device();

  // Assign values
  const int chem_start =
      chemistry_on ? pmbp->pchemistry->get_chemistry_scalars_first_idx() : 0;

  // Optional density spread, in dex about n_H. With the default 0 every cell
  // is identical, which is what makes 4^3 a legitimate accuracy mesh -- but it
  // also makes the substep count identical in every cell, so the problem cannot
  // exhibit warp divergence and every cost measured on it is a divergence-free
  // best case. A non-zero spread ramps n_H log-uniformly along x1, so cells
  // within a warp differ and the adaptive controller has to disagree with
  // itself across a warp, as it would on a real mesh.
  const Real n_H_spread_dex =
      pin->GetOrAddReal("problem", "n_H_spread_dex", 0.0);
  // Scatter the same set of densities across x1 instead of ramping them. The
  // permutation below is a bijection on [0, nx1), so a shuffled run holds the
  // identical multiset of cell densities as the ramp and does the identical
  // total amount of chemical work; only the assignment of cells to warps
  // changes. Cost differences between the two are therefore divergence alone.
  const bool n_H_shuffle =
      pin->GetOrAddBoolean("problem", "n_H_shuffle", false);
  const int nx1_mesh = pmy_mesh_->mesh_indcs.nx1;
  auto &size = pmbp->pmb->mb_size;
  const Real x1min = pmy_mesh_->mesh_size.x1min;
  const Real x1max = pmy_mesh_->mesh_size.x1max;
  const Real mu_H_l = chemistry_on ? pmbp->pchemistry->mu_H
                                   : pin->GetOrAddReal("problem", "mu_H", 1.4);
  const Real mH = pmbp->punit->hydrogen_mass_cgs;
  const Real dens_cgs = pmbp->punit->density_cgs();
  const Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  const Real cs2 = SQR(iso_cs);
  const int nx1_l = indcs.nx1;
  const int is_l = is;

  par_for(
      "pgen_GOW17_hydro", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), ks, ke,
      js, je, is, ie, KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real dens = hydro.d;
        Real eint = hydro.e;
        if (n_H_spread_dex != 0.0) {
          Real &x1minmb = size.d_view(m).x1min;
          Real &x1maxmb = size.d_view(m).x1max;
          const Real x1 = CellCenterX(i - is_l, nx1_l, x1minmb, x1maxmb);
          // -1 at x1min, +1 at x1max, so the spread is symmetric about n_H
          Real f = (x1max > x1min)
                       ? (2.0 * (x1 - x1min) / (x1max - x1min) - 1.0)
                       : 0.0;
          if (n_H_shuffle && nx1_mesh > 1) {
            // Global x1 cell index, recovered from the cell centre.
            const int ig = static_cast<int>(
                (x1 - x1min) / ((x1max - x1min) / nx1_mesh));
            // 37 is prime, so this is a bijection for any nx1 that is not a
            // multiple of 37, and every density in the ramp appears once.
            const int ip = (37 * ig + 11) % nx1_mesh;
            f = 2.0 * static_cast<Real>(ip) / (nx1_mesh - 1) - 1.0;
          }
          const Real n_H_cell = n_H * Kokkos::pow(10.0, n_H_spread_dex * f);
          dens = n_H_cell * mH * mu_H_l / dens_cgs;
          eint = n_H_cell * cs2 / gm1;
        }
        // Assign hydro values to this cell
        w0(m, IDN, k, j, i) = dens;
        w0(m, IVX, k, j, i) = hydro.vx;
        w0(m, IVY, k, j, i) = hydro.vy;
        w0(m, IVZ, k, j, i) = hydro.vz;
        w0(m, IEN, k, j, i) = eint;

        // Assign chemistry values to this cell
        if (chemistry_on) {
          for (size_t s = 0; s < chemistry::GOW17Network::neqs - 1; s++) {
            w0(m, chem_start + s, k, j, i) = initial_chemistry_d(s);
          }
        }
      });

  // Convert primitives to conserved
  pmbp->phydro->peos->PrimToCons(w0, u0, is, ie, js, je, ks, ke);

  return;
}
